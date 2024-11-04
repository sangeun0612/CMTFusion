import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import os
import numpy as np

def get_test_images(paths, height=None, width=None):
    ImageToTensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path)
        image = ImageToTensor(image)
        images.append(image)
    images = torch.stack(images, dim=0)

    return images

def get_image(path):
    image = Image.open(path).convert('RGB')
    return image
