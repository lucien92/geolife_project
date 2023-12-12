#chargeaons nos données
from dataloader import image_loader

#extrait data 1 à 1
train_dataset = image_loader(annotations_file='/mounts/Datasets4/GeoLifeCLEF2022/observations/observations_fr_train.csv', img_dir='patches-fr', transform=None, target_transform=None, tipe='train')

valid_dataset = image_loader(annotations_file='/mounts/Datasets4/GeoLifeCLEF2022/observations/observations_fr_train.csv', img_dir='patches-fr', transform=None, target_transform=None, tipe='val')

from torch.utils.data import DataLoader

#on crée le dataloader

batch_size = 64
shuffle = 'false'

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)


from model import SimpleCNN

# Créez une instance du modèle
num_classes = 17000  # Remplacez par le nombre réel de classes dans vos données
model = SimpleCNN(num_classes)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Définissez votre fonction de perte (loss function) et l'optimiseur Adam.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Vous pouvez ajuster le taux d'apprentissage (lr) selon vos besoins.

# Définissez le nombre d'époques d'entraînement souhaité.
from tqdm import tqdm

# ... [le reste de votre code pour charger les données et définir le modèle] ...

num_epochs = 10

for epoch in range(num_epochs):
    # Mode d'entraînement
    model.train()
    train_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

    for images, labels in train_loader_tqdm:
        images = images.view(-1, 12, 256, 256)  # Fusion des canaux des 4 images

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=(train_loss / (batch_size + 1)))

    # Mode d'évaluation (validation)
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)

            # Obtenez les 30 classes les plus probables pour chaque échantillon
            _, top30_preds = torch.topk(outputs, 30, dim=1)

            # Vérifiez si la vraie étiquette se trouve dans les 30 prédictions les plus probables.
            correct_batch = [(true_label in top30_pred) for true_label, top30_pred in zip(labels, top30_preds)]

            # Calculez le nombre de prédictions correctes dans ce lot.
            total_correct += sum(correct_batch)
            total_samples += len(correct_batch)

    # Calculez la précision de validation.
    accuracy = total_correct / total_samples

    # Affichez les résultats de l'époque.
    print(f'Époque [{epoch + 1}/{num_epochs}], Précision de validation: {accuracy:.2%}')
