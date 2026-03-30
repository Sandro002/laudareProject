from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os as os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import silhouette_score

def tsne(namesFeaturesPath):

    perplessita = [5,15,50,100]
   
    categorie = sorted([d for d in os.listdir(namesFeaturesPath) if os.path.isdir(os.path.join(namesFeaturesPath, d))])
    
    for cat in categorie:
        X_features = np.load(os.path.join(namesFeaturesPath,cat,"features.npy"))
        array_nomi = np.load(os.path.join(namesFeaturesPath,cat,"names.npy"))
        for p in tqdm(perplessita, desc=f"categoria: {cat}"):
            riduttore = TSNE(
                n_components=2, 
                perplexity=p, 
                random_state=42, 
                init='pca', 
                learning_rate='auto'
                )
            X_reduced = riduttore.fit_transform(X_features)
            kmeansPlot("vgg16_regression.pth", cat, X_reduced, array_nomi, p)


def kmeansPlot(nome_modello, categoria, X_ridotto, nomi, perplexity):
    cartella_plots = os.path.join("plots","vgg16",nome_modello, categoria)

    df = pd.DataFrame({
        'Nome_Immagine': nomi,
        'Dim_X': X_ridotto[:, 0],
        'Dim_Y': X_ridotto[:, 1]
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Clustering '{categoria.capitalize()}'", fontsize=18, fontweight='bold', y=1.05)

    for i, k in enumerate([2, 3, 4]):
        clusterer = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = clusterer.fit_predict(X_ridotto)
            
        score = silhouette_score(X_ridotto, labels)
            
        df[f'Colonna{k}'] = labels

        ax = axes[i]
        scatter = ax.scatter(X_ridotto[:, 0], X_ridotto[:, 1], c=labels, cmap='tab10', s=40, edgecolors='white', linewidth=0.5, alpha=0.9)
            
        ax.set_title(f"K = {k} (Score: {score:.3f})", fontsize=12, pad=10)
        ax.axis('off') 
            

        handles, _ = scatter.legend_elements()
        legend_labels = [f"Cluster {c}" for c in range(k)]
        ax.legend(handles, legend_labels, title="Cluster", loc='upper right', fontsize=9)

        plt.tight_layout()
    
    path_plot = os.path.join(cartella_plots,f"plot_perplexity_{perplexity}.png")
    os.makedirs(cartella_plots, exist_ok=True)
    plt.savefig(path_plot, dpi=300, bbox_inches='tight')
    plt.close()


def kmeans(namesFeaturesPath,modelName="vgg16_regression.pth"):

    cartella_plots = os.path.join("plots","vgg16",modelName)
    perplessita = [5,15,50,100]
    categorie = sorted([d for d in os.listdir(namesFeaturesPath) if os.path.isdir(os.path.join(namesFeaturesPath, d))])

    for cat in categorie:
        X_features = np.load(os.path.join(namesFeaturesPath,cat,"features.npy"))
        array_nomi = np.load(os.path.join(namesFeaturesPath,cat,"names.npy"))
        for p in tqdm(perplessita, desc=f"categoria: {cat}"):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f"Clustering '{cat.capitalize()}'", fontsize=18, fontweight='bold', y=1.05)

            for i, k in enumerate([2, 3, 4]):
                clusterer = KMeans(n_clusters=k, random_state=42, n_init="auto")
                labels = clusterer.fit_predict(X_features)
                score = silhouette_score(X_features, labels)

                riduttore = TSNE(
                    n_components=2, 
                    perplexity=p, 
                    random_state=42, 
                    init='pca', 
                    learning_rate='auto'
                )

                X_reduced = riduttore.fit_transform(X_features)
                ax = axes[i]
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10', s=40, edgecolors='white', linewidth=0.5, alpha=0.9)
            
                ax.set_title(f"K = {k} (Score: {score:.3f})", fontsize=12, pad=10)
                ax.axis('off') 
            

                handles, _ = scatter.legend_elements()
                legend_labels = [f"Cluster {c}" for c in range(k)]
                ax.legend(handles, legend_labels, title="Cluster", loc='upper right', fontsize=9)

            plt.tight_layout()
            cartella_plots_cat=os.path.join(cartella_plots,cat)
            path_plot = os.path.join(cartella_plots_cat,f"plot_perplexity_{p}.png")
            os.makedirs(cartella_plots_cat, exist_ok=True)
            plt.savefig(path_plot, dpi=300, bbox_inches='tight')
            plt.close(fig)


if __name__ == "__main__":
    
    kmeans("features/vgg16/default", modelName="default")