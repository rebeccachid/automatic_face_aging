import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import kagglehub

# ----- Téléchargement du dataset UTKFace via KaggleHub -----
# Utilise le dataset "jangedoo/utkface-new"
path = kagglehub.dataset_download("jangedoo/utkface-new")
# On suppose que les images sont extraites dans le dossier "UTKFace"
utkface_dir = os.path.join(path, "UTKFace")  # Ajustez si nécessaire

# ----- Transformation pour l'autoencodeur -----
# Redimensionnement à 224×224 et normalisation pour obtenir des valeurs dans [-1,1]
ae_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ----- Dataset UTKFace -----
# Les fichiers UTKFace sont nommés "age_gender_race_date.jpg".
# On extrait l'âge à partir du premier segment du nom de fichier.
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Répertoire contenant les images UTKFace.
            transform (callable, optionnel): Transformations à appliquer aux images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")
        parts = filename.split('_')
        age = float(parts[0])
        if self.transform:
            image = self.transform(image)
        return image, age

# Création du DataLoader
dataset = UTKFaceDataset(root_dir=utkface_dir, transform=ae_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# ----- Définition d'un autoencodeur simple -----
class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(SimpleAutoencoder, self).__init__()
        # Encodeur : réduit progressivement la taille de l'image
        self.encoder = nn.Sequential(
            # Input: (3, 224, 224) -> Output: (64, 112, 112)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (64, 112, 112) -> (128, 56, 56)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (128, 56, 56) -> (256, 28, 28)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (256, 28, 28) -> (latent_dim, 14, 14)
            nn.Conv2d(256, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Décodeur : reconstruit l'image à partir de la représentation latente
        self.decoder = nn.Sequential(
            # (latent_dim, 14, 14) -> (256, 28, 28)
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (256, 28, 28) -> (128, 56, 56)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (128, 56, 56) -> (64, 112, 112)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (64, 112, 112) -> (3, 224, 224)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Sortie dans [-1, 1]
        )
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out

# ----- Entraînement -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleAutoencoder(latent_dim=256).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

# Sauvegarde du modèle entraîné
torch.save(model.state_dict(), "simple_autoencoder_utkface.pth")
print("Modèle sauvegardé sous 'simple_autoencoder_utkface.pth'")
