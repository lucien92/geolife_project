import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch

class image_loader(Dataset):
    def __init__(self, annotations_file='/mounts/Datasets4/GeoLifeCLEF2022/observations/observations_fr_train.csv', img_dir='patches-fr', transform=None, target_transform=None, tipe = 'train'):
        self.img_labels = pd.read_csv(annotations_file, delimiter=';')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = transforms.ToTensor()
        self.tipe = tipe

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        idx += 1  # Commencer l'index à 1 au lieu de 0

        base_path = '/mounts/Datasets4/GeoLifeCLEF2022/patches-fr/'
        while True:  # Boucle infinie pour continuer l'itération jusqu'à ce qu'une ligne "train" soit trouvée
            observation_id = str(self.img_labels.iloc[idx, 0])

            if self.img_labels.iloc[idx, 4] == self.tipe:
                break  # Sortir de la boucle si la ligne correspond à "train"
            else:
                idx += 1  

        images_tensors = []
        for image_suffix in ['_rgb.jpg', '_near_ir.jpg', '_landcover.tif', '_altitude.tif']:
            img_path = os.path.join(base_path, observation_id[-2:], observation_id[-4:-2], observation_id + image_suffix)

            if os.path.exists(img_path):
                image = Image.open(img_path)

                #on convertit les mono canaux en multi canaux
                if image.mode != 'RGB':
                    image = image.convert('RGB')  # Convertir en RGB si ce n'est pas déjà le cas, car sinon problème car tenseur de taille différentes

                if self.transform:
                    image = self.transform(image)
                image_tensor = self.to_tensor(image)
                images_tensors.append(image_tensor)
            else:
                print(f"Image manquante pour l'ID {observation_id}")
                return torch.zeros(4, 3, 256, 256), -1  # Retourne un tenseur vide et un label -1 en cas d'image manquante

        if len(images_tensors) == 4:
            images_tensor = torch.stack(images_tensors, dim=0)  # Empile les tenseurs le long d'une nouvelle dimension
        else:
            return torch.zeros(4, 3, 256, 256), -1

        label = self.img_labels.loc[self.img_labels['observation_id'] == int(observation_id), 'species_id'].values[0]
        if self.target_transform:
            label = self.target_transform(label)

        return images_tensor, label

# #On teste que le dataloader marche bien

# # Transformation pour redimensionner les images à une taille spécifique (par exemple 256x256)
# data_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     # Ajoute d'autres transformations si nécessaire
# ])

# # Création de l'instance du DataLoader
# dataset = image_loader(transform=data_transform) #donc maintenant on fait du resize et du to_tensor()
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Utilisation du DataLoader pour charger les données
# for batch_idx, (images, label) in enumerate(dataloader):
#     if images is not None:
#         print(f"Batch {batch_idx + 1}:")  # Correction pour afficher le batch à partir de 1
#         for i in range(len(images)):
#             print(f"Image {i+1} - Shape: {images[i].size()}")
#         print(f"Label: {label}")

