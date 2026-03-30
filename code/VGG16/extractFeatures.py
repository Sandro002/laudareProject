from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import torch as torch
from torch import nn
import glob
import os
import numpy as np
import sys
from vgg16_utilities import *


def extractFeatures(transform, path_modello, nome, cartella_base="dataset", cartella_salvataggio="features", ):

    path_modello= path_modello
    device_temp = torch.device('cpu')
    if nome == None:
        modello = getDefaultModel()
    else:
        path_modello = os.path.join(path_modello, nome) 
        print(path_modello)
        state_dict = torch.load(path_modello, map_location=device_temp, weights_only=True)
        modello = models.vgg16()

        num_classi_allenate = state_dict['classifier.6.weight'].shape[0]
        modello.classifier[6] = nn.Linear(4096, num_classi_allenate)

        modello.load_state_dict(state_dict)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    modello = modello.to(device)


    categorie = sorted([d for d in os.listdir(cartella_base) if os.path.isdir(os.path.join(cartella_base, d))])

    if not categorie:
        print(f"Nessuna categoria trovata in '{cartella_base}'.")
        return None, None

    print(f"\nCategorie disponibili:")
    for i, cat in enumerate(categorie):
        
        cartella_target = os.path.join(cartella_base, cat)

        #preparazione modello

        modello.classifier[6] = nn.Identity()

        modello = modello.to(device)
        modello.eval()

        #estrazione dati
        features_lista = []
        nomi_file_lista = []
        
        lista_file_paths = glob.glob(f"{cartella_target}/**/*.*", recursive=True)
        lista_file_paths = [f for f in lista_file_paths if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        with torch.no_grad():
            for path_completo in tqdm(lista_file_paths, desc="Estrazione"):
                nome_file = os.path.basename(path_completo)
                try:
                    img = Image.open(path_completo).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    vettore_features = modello(img_tensor)
                    vettore_numpy = vettore_features.cpu().numpy().flatten()
                        
                    features_lista.append(vettore_numpy)
                    nomi_file_lista.append(nome_file)
                except Exception as e:
                    print(f"\nErrore su {nome_file}: {e}")
                    

        #salvataggio
        if not features_lista:
            print("Nessuna feature estratta.")
            return None, None

        X_features = np.array(features_lista)
        if path_modello == None:
            sottocartella = os.path.join(cartella_salvataggio, "vgg16", "default", cat)
        else:
            sottocartella = os.path.join(cartella_salvataggio, "vgg16", nome, cat)
        os.makedirs(sottocartella, exist_ok=True)
        
        path_features = os.path.join(sottocartella, "features.npy")
        path_nomi = os.path.join(sottocartella, "names.npy")

        np.save(path_features, X_features)
        np.save(path_nomi, np.array(nomi_file_lista))

        print(f"Completato {X_features.shape[0]} features salvate in:\n{sottocartella}")


if __name__ == "__main__":
    
    VGG_REGRESSION="vgg16_regression.pth"
    extractFeatures(getTransformVgg16(), "models/vgg16", VGG_REGRESSION)