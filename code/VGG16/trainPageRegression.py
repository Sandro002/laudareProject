from vgg16_utilities import *
from PIL import Image
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


    
class PageRegressionDataset(Dataset):
    def __init__(self, rootDir="dataset", transform=getTransformVgg16()):
        self.rootDir = rootDir
        self.transform = transform
        self.imgPath = []
        self.pageNames = []

        categorie = getCategories()

        pageSet= set()
        for category in os.listdir(rootDir):

            if category.lower() not in categorie:
                continue

            categoriyPath = os.path.join(rootDir, category)
            
            if os.path.isdir(categoriyPath):
                for file in os.listdir(categoriyPath):
                    if file.endswith(('.png','.jpg')):
                        pageName = file.split('_id')[0]
                        self.imgPath.append(os.path.join(categoriyPath,file))
                        self.pageNames.append(pageName)
                        pageSet.add(pageName)

        sortedPages= sorted(list(pageSet))
        
        numPages = len(sortedPages)
        self.normalizedPages = {}
        for id, page in enumerate(sortedPages): #normalizzazione pagine da 0 a 1 -> numPag/totPage
            self.normalizedPages[page] = id / (numPages-1)
        self.labels = [self.normalizedPages[p] for p in self.pageNames]
        print(self.labels)
    def __len__(self):
        return len(self.imgPath)
        
    def __getitem__(self, index):
        image=Image.open(self.imgPath[index]).convert('RGB')
        labelTensor = torch.tensor([self.labels[index]], dtype=torch.float32)
        image = self.transform(image)

        return image, labelTensor



def trainPages(dataset_folder="dataset", num_epoch=15, learning_rate=0.0001, batch_size=32):

    #carico il modello di default e assegno un fully connected layer: input -> num features, output -> 1 numero scalare
    modello =getDefaultModel()
    num_ftrs = modello.classifier[6].in_features
    modello.classifier[6] = nn.Linear(num_ftrs, 1)

    device=get_device()
    lossFunction= nn.L1Loss()

    dataset = PageRegressionDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=3)
    print("Inizio Fine - Tuning")

    modello=modello.to(device)

    optimizer = torch.optim.Adam(modello.parameters(), lr=learning_rate)
   
    for epoca in range(num_epoch):
        modello.train()
        currentLoss = 0.0
        
        for immagini, labels in tqdm(dataloader, desc=f"Epoca {epoca+1}/{num_epoch}", total=len(dataloader)):
            
            immagini = immagini.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = modello(immagini)
            loss = lossFunction(output, labels)
            loss.backward()
            optimizer.step()

            currentLoss += loss.item()

        epoch_loss = currentLoss / len(dataloader)
        print(f"loss rate per epoca {epoca}: {epoch_loss}")

    modello.eval()
    modello=modello.to("cpu")

    return modello
    
    
MODELPATH =  "models/vgg16"
MODELNAME = "vgg16_regression.pth"


if __name__ == "__main__":
    

    modelFinetuned = trainPages()

    os.makedirs(MODELPATH, exist_ok=True)
    torch.save(modelFinetuned.state_dict(), os.path.join(MODELPATH,MODELNAME))

    print("Modello allenato e salvato in: ", os.path.join(MODELPATH,MODELNAME))