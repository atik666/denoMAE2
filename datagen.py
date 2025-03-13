import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
from pathlib import Path

class DenoMAEDataGenerator(Dataset):
    def __init__(self, noisy_image_path, noiseless_img_path, image_size=(224, 224), transform=None):
        """
        Initializes the data generator for DenoMAE.
        
        Args:
            noisy_image_path (str): Directory containing noisy image files (e.g., PNG).
            noiseLess_image_path (str): Directory containing noiseless image files (e.g., PNG).
            image_size (tuple): Size to which images will be resized.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_size = image_size
        self.transform = transform or transforms.ToTensor()

        # Use pathlib for more Pythonic path handling
        self.paths = {
            'noisy_image': Path(noisy_image_path),
            'noiseless_image': Path(noiseless_img_path),
        }
        
        # Use list comprehension for file listing
        self.filenames = {
            key: sorted(path.glob('*')) for key, path in self.paths.items()
        }

    def __len__(self):
        # Return the length of the largest list of filenames
        return max(len(filenames) for filenames in self.filenames.values())

    def __getitem__(self, index):
        # Load and preprocess image
        noisy_img = Image.open(self.filenames['noisy_image'][index]).resize(self.image_size)
        noisy_img = self.transform(noisy_img)

        noiseless_img = Image.open(self.filenames['noiseless_image'][index]).resize(self.image_size)
        noiseless_img = self.transform(noiseless_img)
        
        return noisy_img, noiseless_img

if __name__ == "__main__":
    DenoMAEDataGenerator