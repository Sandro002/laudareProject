from torchvision import models,transforms
import os
import torch

def getTransformVgg16():

    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    return transform

def getDefaultModel():
    return models.vgg16(weights=models.VGG16_Weights.DEFAULT)

def getCategories(cartella_base="dataset"):
    categorie = sorted([d for d in os.listdir(cartella_base) if os.path.isdir(os.path.join(cartella_base, d))])

    return categorie

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")