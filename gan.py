import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import kagglehub

# ---------------------------
# 1. Chargement du dataset UTKFace via KaggleHub
# ---------------------------
path = kagglehub.dataset_download("jangedoo/utkface-new")
utkface_dir = os.path.join(path, "UTKFace")  # Ajustez selon la structure du dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png'))]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")
        # On suppose que le nom de fichier commence par l'âge (ex: "25_...")
        age = float(filename.split('_')[0])
        if self.transform:
            image = self.transform(image)
        return image, age

dataset = UTKFaceDataset(root_dir=utkface_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------------------
# 2. Préparation de la condition d'âge
# ---------------------------
c_dim = 10  # Nombre de classes d'âge (exemple : 10 classes)
def one_hot(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels]

# ---------------------------
# 3. Définition des architectures (Generator, Discriminator et ResidualBlock)
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim, affine=True)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=10, repeat_num=6):
        super().__init__()
        self.c_dim = c_dim
        # Concaténation de l'image et du vecteur one-hot en entrée
        self.initial = nn.Sequential(
            nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, padding=3),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = conv_dim
        down_layers = []
        for _ in range(2):
            down_layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2
        self.down = nn.Sequential(*down_layers)
        self.res_blocks = nn.Sequential(*[ResidualBlock(curr_dim) for _ in range(repeat_num)])
        up_layers = []
        for _ in range(2):
            up_layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim // 2, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim //= 2
        self.up = nn.Sequential(*up_layers)
        self.final = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )
    def forward(self, x, c):
        B, _, H, W = x.size()
        c = c.view(B, self.c_dim, 1, 1).expand(B, self.c_dim, H, W)
        x = torch.cat([x, c], dim=1)
        x = self.initial(x)
        x = self.down(x)
        x = self.res_blocks(x)
        x = self.up(x)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self, img_size=224, conv_dim=64, c_dim=10):
        super().__init__()
        layers = [nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01)]
        curr_dim = conv_dim
        for _ in range(1, 6):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.01)
            ]
            curr_dim *= 2
        self.main = nn.Sequential(*layers)
        kernel_size = img_size // (2 ** 6)
        self.conv_src = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size)
        self.conv_cls = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size)
    def forward(self, x):
        out = self.main(x)
        out_src = self.conv_src(out)
        out_cls = self.conv_cls(out)
        return out_src.view(x.size(0), -1), out_cls.view(x.size(0), -1)

# ---------------------------
# 4. Classifieur pré-entraîné pour guidance d'âge
# ---------------------------
class ResNetMultiTask(nn.Module):
    def __init__(self):
        super(ResNetMultiTask, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # (B, 2048)
        self.age_regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        # La branche genre est définie mais non utilisée ici
        self.gender_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        features = self.backbone(x)
        age = self.age_regressor(features)
        gender = self.gender_classifier(features)
        return age, gender

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = ResNetMultiTask().to(device)
classifier.load_state_dict(torch.load("resnet_multitask_utkface.pth", map_location=device))
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False

def ae_to_classifier(x):
    # Conversion de l'image de [-1,1] à [0,1] et normalisation selon ImageNet
    x = (x + 1) / 2
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std

# Représentations d'âges pour chaque classe (pour la guidance)
rep_ages = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], dtype=torch.float32, device=device)

# ---------------------------
# 5. Initialisation des modèles et optimisateurs
# ---------------------------
start_epoch = 0

G = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)
if start_epoch != 0:
    G.load_state_dict(torch.load(f"saved_models/generator_guided_epoch_{start_epoch}.pth", map_location=device))
D = Discriminator(img_size=224, conv_dim=64, c_dim=c_dim).to(device)
if start_epoch != 0:
    D.load_state_dict(torch.load(f"saved_models/discriminator_guided_epoch_{start_epoch}.pth", map_location=device))
optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

adv_loss_fn = nn.BCEWithLogitsLoss()
cls_loss_fn = nn.CrossEntropyLoss()
cycle_loss_fn = nn.L1Loss()
mse_loss_fn = nn.MSELoss()

lambda_cycle = 50.0
lambda_cls_guidance = 0.5

# Création du dossier pour sauvegarder les images générées et les modèles si besoin
if not os.path.exists("fake_images"):
    os.makedirs("fake_images")
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

# ---------------------------
# 6. Boucle d'entraînement
# ---------------------------
num_epochs = 40
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

for epoch in range(start_epoch, num_epochs):
    G.train()
    D.train()
    total_G_loss = 0.0
    total_D_loss = 0.0
    last_fake = None  # Pour stocker la dernière fake_images
    for images, true_ages in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        batch_size = images.size(0)
        # Discrétisation de l'âge source
        src_labels = (true_ages / (100.0 / c_dim)).long().to(device)
        src_labels = torch.clamp(src_labels, 0, c_dim - 1)
        src_onehot = one_hot(src_labels, c_dim)
        # Sélection de labels cibles différents pour modifier l'âge
        target_labels = []
        for lbl in src_labels:
            choices = list(range(c_dim))
            choices.remove(lbl.item())
            target_labels.append(random.choice(choices))
        target_labels = torch.tensor(target_labels, device=device)
        target_onehot = one_hot(target_labels, c_dim)
        
        # --- Entraînement du Discriminateur ---
        optimizer_D.zero_grad()
        real_valid = torch.ones(batch_size, 1, device=device)
        fake_valid = torch.zeros(batch_size, 1, device=device)
        
        out_src, out_cls = D(images)
        loss_D_adv_real = adv_loss_fn(out_src, real_valid)
        loss_D_cls = cls_loss_fn(out_cls, src_labels)
        
        fake_images = G(images, target_onehot)
        out_src_fake, _ = D(fake_images.detach())
        loss_D_adv_fake = adv_loss_fn(out_src_fake, fake_valid)
        
        loss_D = 0.5 * (loss_D_adv_real + loss_D_adv_fake) + loss_D_cls
        loss_D.backward()
        optimizer_D.step()
        
        # --- Entraînement du Générateur --- 
        optimizer_G.zero_grad()
        fake_images = G(images, target_onehot)

        out_src_fake, out_cls_fake = D(fake_images)
        loss_G_adv = adv_loss_fn(out_src_fake, real_valid)
        loss_G_cls = cls_loss_fn(out_cls_fake, target_labels)
        # Consistance cyclique pour préserver l'identité
        rec_images = G(fake_images, src_onehot)
        loss_cycle = cycle_loss_fn(rec_images, images)
        # Guidance par le classifieur pour contraindre l'âge généré
        fake_for_cls = ae_to_classifier(fake_images)
        pred_age, _ = classifier(fake_for_cls)
        rep_age = rep_ages[target_labels].unsqueeze(1)
        loss_age_guidance = mse_loss_fn(pred_age, rep_age)
        
        loss_G = loss_G_adv + loss_G_cls + lambda_cycle * loss_cycle + lambda_cls_guidance * loss_age_guidance
        loss_G.backward()
        optimizer_G.step()
                
        total_G_loss += loss_G.item() * batch_size
        total_D_loss += loss_D.item() * batch_size
        
    avg_G_loss = total_G_loss / len(dataset)
    avg_D_loss = total_D_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - G Loss: {avg_G_loss:.4f} | D Loss: {avg_D_loss:.4f}")
    
    # Enregistrement de la dernière fake_images à la fin de l'époque
    last_fake = fake_images.detach()
    if last_fake is not None:
        # On prend la première image du batch et on la convertit en image PIL.
        # On passe de [-1,1] à [0,1] en faisant (img + 1) / 2
        img_pil = to_pil((last_fake[0] + 1) / 2)
        img_pil.save(f"fake_images/fake_epoch_{epoch+1}.png")
    
    # Sauvegarde des modèles tous les 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(G.state_dict(), f"saved_models/generator_guided_epoch_{epoch+1}.pth")
        torch.save(D.state_dict(), f"saved_models/discriminator_guided_epoch_{epoch+1}.pth")
        print(f"Modèles sauvegardés pour l'epoch {epoch+1}.")

# Sauvegarde finale des modèles
torch.save(G.state_dict(), "saved_models/generator_guided_final.pth")
torch.save(D.state_dict(), "saved_models/discriminator_guided_final.pth")
print("Modèles sauvegardés définitivement.")
