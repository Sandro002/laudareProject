import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as npy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

#classi
class Annotation:
    def __init__(self, d_json, catName):
        
        #dati boundbox
        bbox =   d_json['bbox']
        self.x = int(bbox[0])
        self.y = int(bbox[1])
        self.w = int(bbox[2])
        self.h = int(bbox[3])
        
        #dati identificativi
        self.id = d_json['id']
        self.category = catName 
        


    def __repr__(self):
        return f"<Annotazione {self.id}: {self.category} a ({self.x},{self.y})>"


class Page:
    def __init__(self, data_json, image_folder):
        self.id = data_json['id']
        self.filename = data_json['file_name']
        self.width = data_json['width']
        self.height = data_json['height']
        self.full_path = os.path.join(image_folder, self.filename)
        
        self.annotations = [] 
        

    def add_annotation(self, ann):
         self.annotations.append(ann)

    def __repr__(self):
        return f"<Pagina id:{self.id} file:{self.filename} - {len(self.annotations)} annotazioni>"


#leggo gt.json

CATEGORIES={
    1 : 'neume',
    2 : 'clef',
    3 : 'custos',
    6 : 'lines',
}

print(f"Categorie da analizzare: {CATEGORIES}")

img_folder='I-Fn_BR_18'

with open('I-Fn_BR_18\\annotations-diplomatic\\gt.json') as f:
    d = json.load(f)

pages_dict = {}

#creo un array di pagine vuote

for img_data in d['images']:
        p = Page(img_data, img_folder)
        pages_dict[p.id] = p

#riempio le pagine con le annotazioni che vogliamo osservare
for page in pages_dict.values():
     for annotation in d['annotations']:
          if int (annotation['image_id']) == page.id and annotation['category_id'] in CATEGORIES:
               page.add_annotation(Annotation(annotation, annotation['category_id']))




#ritaglio boundingbox

print('Applicando BoundingBox alle immagini...\n')

OUTPUT_DIR = 'images_crop' #cartella delle immagini croppate, se non esiste la creo
error=[]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for page in tqdm(pages_dict.values()):
    imagePath = page.full_path
    try:
          with Image.open(imagePath) as img:
               
               for ann in page.annotations:
                    #taglio immagine con metodo crop(X_min, Y_min, X_max, Y_max)
                    bbox= (ann.x, ann.y, ann.x + ann.w, ann.y + ann.h)
                    imgCrop= img.crop(bbox)
                    
                    #immagini ritagliate vengono salvate in cartella 'imgCrop' -> sottocartella con nome categoria

                    catDir = os.path.join(OUTPUT_DIR, CATEGORIES[ann.category])

                    #nome file nome pagina_nome categoria es p1_12.jpg

                    filename = f"p{page.id}_{ann.id}.jpg"
                    savePath = os.path.join(catDir, filename)

                    if not os.path.exists(catDir):
                         os.makedirs(catDir)

                    imgCrop.save(savePath)
    except FileNotFoundError :
        error.append(page.id)

print(f"errore per i file con id {error}")


#carico vgg16 con i pesi pre-allenati

weights= models.VGG16_Weights.IMAGENET1K_V1
model=models.vgg16(weights=weights)

#assegno una funzione identitè all'ultomo layer del modello 

model.classifier[6] = nn.Identity() 

model.eval()

print("\nVGG16 caricato correttamente\n")


transform=transforms.Compose([
     
    #ridimensiono l'immagine a un quadrato 224x224
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),

    #normalizzo i colori secondo media e deviazione standard di imageNet   
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#carico le immaginia blocchi di 32 
dataset = datasets.ImageFolder(root=OUTPUT_DIR, transform=transform)

dataLoader= DataLoader(dataset, batch_size=32, shuffle=False) 



#utilizzo CUDA se disponibile una gpu NVIDIA
if torch.cuda.is_available(): 
     device=torch.device("cuda")
     print(f"Utilizzando GPU NVIDIA: {torch.cuda.get_device_name(0)}")
#Utilizzo Metal se disponibile un chip Silicon 
elif torch.backend.mps.is_available():
     device=torch.device('mps')
     print(f"Utilizzando accelerazione Metal Apple Silicon")
else:
     #uso la CPU se non è disponibile alcuna accelerazione grafica
     device = torch.device("cpu")
     print("nessuna accelerazione grafica trovata, uso CPU")

model = model.to(device)

featuresList = []
labelsList = []

print("Inizio estrazione features...")

with torch.no_grad(): 
    for images, labels in tqdm(dataLoader):
        
        #sposto le immagini su device
        images= images.to(device)
        
        #estrazione features
        output = model(images)
        
        #salvo le features e le label in liste
        featuresList.append(output.cpu().numpy())
        labelsList.append(labels.numpy())


featuresVgg = npy.concatenate(featuresList)

print(f"estrazine completata, matrice features: {featuresVgg.shape}") 




try:
    labelsVgg = npy.concatenate(labelsList)
except ValueError:
    temp_list = []
    #se labelList è un insieme di numeri e lettere
    for item in labelsList:
        if hasattr(item, '__iter__') and not isinstance(item, str):
            #se è una lsita prendo il contenuto
            temp_list.extend(item)
        else:
            #se è un numero singolo lo aggiungo
            temp_list.append(item) 
    labelsVgg = npy.array(temp_list)

#traduco gli id in categories per farli corrispondere con le Labels
sorted_ids = sorted(CATEGORIES.keys())

names_list = [CATEGORIES[k] for k in sorted_ids]
labelNames = [names_list[int(i)] for i in labelsVgg]


perplexities=[5,30,50,100]

#creo la base per il plot
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle(f"Analisi t-SNE su Features VGG16",fontsize=24)
axes = axes.flatten() 

try:
    features_matrix = npy.vstack(featuresList)
except ValueError:
    features_matrix = npy.concatenate(featuresList, axis=0)


for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(features_matrix)

    plot= pd.DataFrame({
        'x': tsne_results[:,0],
        'y': tsne_results[:,1],
        'Categoria': labelNames
    })
    sns.scatterplot(
        data=plot,
        x='x', y='y',
        hue='Categoria',
        palette='tab10',
        s=60,
        alpha=0.8,
        ax=axes[i],
        legend='full' if i == 0 else False
    )
    axes[i].set_title(f"Perplexity: {perplexity}", fontsize=18)
    axes[i].set_axis_off()
        
    axes[i].set_title(f"Perplexity: {perplexity}", fontsize=18)
    axes[i].set_axis_off()
    

plt.tight_layout()
plt.subplots_adjust(top=0.92)
RESULTSFOLDER="results"
outputFilename = "tsne_vgg16.png"
plt.savefig(os.path.join(RESULTSFOLDER, outputFilename), dpi=300)