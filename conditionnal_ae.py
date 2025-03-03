import os
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
utkface_dir = os.path.join(path, "UTKFace")  # Ajustez selon la structure

# Transformation pour l'autoencodeur : images 224x224, valeurs dans [-1,1]
ae_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset UTKFace (les images sont nommées "age_gender_race_date.jpg")
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg','.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")
        # Extraction de l'âge (premier élément du nom)
        parts = filename.split('_')
        age = float(parts[0])
        if self.transform:
            image = self.transform(image)
        return image, age

# Pour limiter la consommation mémoire : batch_size 4 et num_workers 0
loader = DataLoader(UTKFaceDataset(root_dir=utkface_dir, transform=ae_transform),
                    batch_size=4, shuffle=True, num_workers=0)

# ---------------------------
# 2. Simple Autoencodeur pré-entraîné
# ---------------------------
class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # (64,112,112)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # (128,56,56)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256,28,28)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim, kernel_size=4, stride=2, padding=1),  # (latent_dim,14,14)
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # (256,28,28)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),           # (128,56,56)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),            # (64,112,112)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),              # (3,224,224)
            nn.Tanh()  # valeurs dans [-1,1]
        )
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simple_ae = SimpleAutoencoder(latent_dim=256).to(device)
pretrained_path = "simple_autoencoder_utkface.pth"
if os.path.exists(pretrained_path):
    simple_ae.load_state_dict(torch.load(pretrained_path, map_location=device))
    print("Simple autoencodeur pré-entraîné chargé.")
else:
    print("Fichier simple_autoencoder_utkface.pth introuvable, entraînement à partir de zéro.")

# ---------------------------
# 3. Modèle Conditionnel basé sur le simple autoencodeur pré-entraîné
# ---------------------------
class ConditionalAutoencoderFromPretrained(nn.Module):
    def __init__(self, simple_ae, latent_dim=256, spatial_size=14):
        super(ConditionalAutoencoderFromPretrained, self).__init__()
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        # On conserve l'encodeur et le décodeur du simple AE pré-entraîné
        self.encoder = simple_ae.encoder
        self.decoder = simple_ae.decoder
        # Branche d'embedding pour la condition d'âge
        self.age_embedding = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.ReLU(inplace=True)
        )
        # Fusion : concaténer la représentation latente aplatie et l'embedding d'âge
        self.fusion_fc = nn.Sequential(
            nn.Linear(latent_dim * spatial_size * spatial_size + latent_dim,
                      latent_dim * spatial_size * spatial_size),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, target_age):
        B = x.size(0)
        latent = self.encoder(x)  # (B, latent_dim, spatial_size, spatial_size)
        flat_latent = latent.view(B, -1)  # (B, latent_dim * spatial_size * spatial_size)
        age_emb = self.age_embedding(target_age.float())  # (B, latent_dim)
        combined = torch.cat([flat_latent, age_emb], dim=1)  # (B, latent_dim*spatial_size*spatial_size + latent_dim)
        fused = self.fusion_fc(combined)  # (B, latent_dim*spatial_size*spatial_size)
        fused_latent = fused.view(B, self.latent_dim, self.spatial_size, self.spatial_size)
        out = self.decoder(fused_latent)
        return out

conditional_ae = ConditionalAutoencoderFromPretrained(simple_ae, latent_dim=256, spatial_size=14).to(device)

# ---------------------------
# 4. Chargement du classifieur fine-tuné (ResNet multi-tâche)
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

classifier = ResNetMultiTask().to(device)
classifier.load_state_dict(torch.load("resnet_multitask_utkface.pth", map_location=device))
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False

def ae_to_classifier(x):
    x = (x + 1) / 2  # Convertir de [-1,1] à [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
    return (x - mean) / std

# ---------------------------
# 5. Fine-tuning du modèle conditionnel avec supervision conditionnelle (50% reconstruction, 50% transformation)
# ---------------------------
optimizer = optim.Adam(conditional_ae.parameters(), lr=1e-4)
criterion_recon = nn.MSELoss()  # Loss de reconstruction
criterion_cls = nn.MSELoss()    # Loss de transformation via le classifieur

num_epochs = 20
lambda_recon = 500.0
lambda_cls = 1.0

# Utilisation de la précision mixte
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

for epoch in range(num_epochs):
    conditional_ae.train()
    running_loss = 0.0
    running_recon_loss = 0.0
    running_cls_loss = 0.0
    for images, true_ages in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        true_ages = true_ages.to(device).unsqueeze(1).float()  # (B, 1)
        batch_size = images.size(0)
        # Création d'un masque aléatoire : 50% reconstruction (mask=True), 50% transformation (mask=False)
        mask = torch.rand(batch_size, device=device) < 0.5
        target_ages = true_ages.clone()
        # Pour les cas de transformation, tirer un âge cible aléatoire (moyenne=50, std=25, clampé entre 0 et 100)
        rand_ages = torch.clamp(torch.normal(mean=50.0, std=25.0, size=true_ages.shape, device=device, dtype=true_ages.dtype), 0, 100)
        target_ages[~mask] = rand_ages[~mask]
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = conditional_ae(images, target_ages)
            # Calcul de la loss de reconstruction
            if mask.sum() > 0:
                loss_recon = criterion_recon(output[mask], images[mask])
            else:
                loss_recon = 0.0 * output.sum()
            # Calcul de la loss de transformation
            if (~mask).sum() > 0:
                output_for_cls = ae_to_classifier(output[~mask])
                pred_age = classifier(output_for_cls)[0].squeeze()  # Branche d'âge
                loss_cls = criterion_cls(pred_age, target_ages[~mask].squeeze())
            else:
                loss_cls = 0.0 * output.sum()
            loss = lambda_recon * loss_recon + lambda_cls * loss_cls
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_recon_loss += loss_recon.item() * images.size(0)
        running_cls_loss += loss_cls.item() * images.size(0)
        
        torch.cuda.empty_cache()
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_recon_loss = running_recon_loss / len(loader.dataset)
    epoch_cls_loss = running_cls_loss / len(loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Combined Loss: {epoch_loss:.4f} | Recon: {lambda_recon*epoch_recon_loss:.4f} | Transform: {lambda_cls*epoch_cls_loss:.4f}")
    torch.cuda.empty_cache()

torch.save(conditional_ae.state_dict(), "conditional_autoencoder_finetuned_utkface.pth")
print("Modèle conditionnel fine-tuné sauvegardé sous 'conditional_autoencoder_finetuned_utkface.pth'")
