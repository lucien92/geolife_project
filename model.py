import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # La première couche convolutive prend une image à 12 canaux en entrée
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # La taille de l'entrée pour la première couche entièrement connectée doit être ajustée
        # en fonction de la taille réduite de l'image après les couches convolutives et de pooling
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x est maintenant de la forme [batch_size * 4, 12, 256, 256]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Redimensionnement pour la couche entièrement connectée
        x = x.view(x.size(0), -1)  # Aplatir les caractéristiques pour la couche entièrement connectée

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
