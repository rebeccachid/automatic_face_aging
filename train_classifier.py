import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import kagglehub
from tqdm import tqdm

# ----- Téléchargement et chargement du dataset UTKFace via KaggleHub -----
# On utilise par exemple le dataset "jangedoo/utkface-new"
path = kagglehub.dataset_download("jangedoo/utkface-new")
# Supposons que les images se trouvent dans le dossier "UTKFace"
utkface_dir = os.path.join(path, "UTKFace")  # Ajustez selon la structure

# Transformation d'entraînement et de validation (normalisation ImageNet)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset personnalisé pour UTKFace
# Les fichiers sont nommés "age_gender_race_date.jpg"
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Dossier contenant toutes les images UTKFace.
            transform (callable, optionnel): Transformations à appliquer.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Lister les fichiers jpg et png
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg','.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")
        # Extraire l'âge et le genre depuis le nom de fichier
        # Format attendu: "age_gender_race_date.jpg"
        parts = filename.split('_')
        age = float(parts[0])
        gender = int(parts[1])  # 0: Male, 1: Female
        if self.transform:
            image = self.transform(image)
        return image, age, gender

# Chargement du dataset complet
full_dataset = UTKFaceDataset(root_dir=utkface_dir, transform=train_transform)
# Séparation en jeux d'entraînement (80%) et validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# Pour validation, utiliser les transformations de validation
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ----- Définition du modèle multi-tâche basé sur ResNet50 -----
class ResNetMultiTask(nn.Module):
    def __init__(self):
        super(ResNetMultiTask, self).__init__()
        # Charger ResNet50 pré-entraîné et remplacer la dernière couche
        self.backbone = models.resnet50(pretrained=True)
        # On remplace la couche fc par une identité pour obtenir le vecteur de features
        self.backbone.fc = nn.Identity()  # La sortie aura 2048 dimensions
        
        # Tête pour la régression d'âge
        self.age_regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        # Tête pour la classification du genre (2 classes)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)  # (B, 2048)
        age = self.age_regressor(features)       # (B, 1)
        gender = self.gender_classifier(features)  # (B, 2)
        return age, gender

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetMultiTask().to(device)

# ----- Définition de la fonction de perte, optimiseur et scheduler -----
criterion_age = nn.MSELoss()             # Pour la régression d'âge
criterion_gender = nn.CrossEntropyLoss()   # Pour la classification du genre

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs = 20

# ----- Entraînement -----
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_age_loss = 0.0
    running_gender_loss = 0.0
    running_abs_error = 0.0
    for images, ages, genders in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        ages = ages.to(device).unsqueeze(1).float()  # (B, 1)
        genders = genders.to(device).long()           # (B,)
        
        optimizer.zero_grad()
        age_pred, gender_pred = model(images)
        
        loss_age = criterion_age(age_pred, ages)
        loss_gender = criterion_gender(gender_pred, genders)
        loss = loss_age + loss_gender
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_age_loss += loss_age.item() * images.size(0)
        running_gender_loss += loss_gender.item() * images.size(0)
        running_abs_error += torch.sum(torch.abs(age_pred - ages)).item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_age_loss = running_age_loss / len(train_loader.dataset)
    epoch_gender_loss = running_gender_loss / len(train_loader.dataset)
    train_mae = running_abs_error / len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_abs_error = 0.0
    val_samples = 0
    with torch.no_grad():
        for images, ages, genders in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            ages = ages.to(device).unsqueeze(1).float()
            genders = genders.to(device).long()
            age_pred, gender_pred = model(images)
            
            loss_age = criterion_age(age_pred, ages)
            loss_gender = criterion_gender(gender_pred, genders)
            loss_val = loss_age + loss_gender
            
            val_loss += loss_val.item() * images.size(0)
            val_abs_error += torch.sum(torch.abs(age_pred - ages)).item()
            val_samples += images.size(0)
    val_loss /= val_samples
    val_mae = val_abs_error / val_samples
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {epoch_loss:.4f} (Age: {epoch_age_loss:.4f}, Gender: {epoch_gender_loss:.4f}), Train MAE: {train_mae:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

# Sauvegarde du modèle fine-tuné
torch.save(model.state_dict(), "resnet_multitask_utkface.pth")
print("Modèle sauvegardé sous 'resnet_multitask_utkface.pth'")
