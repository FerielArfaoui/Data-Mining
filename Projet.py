
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import preprocessing
# Charger les données depuis le fichier fromage.txt
filename = 'fromage.txt'
fromage = pd.read_csv(filename, sep=r'\s+', index_col=0)
# Retirer les observations du groupe n°0 du k-means précédent
fromage_subset = fromage.iloc[kmeans.labels_!=0,:]
# Centrer et réduire
fromage_subset_cr = preprocessing.scale(fromage_subset)
# Générer la matrice des liens
Z_subset = linkage(fromage_subset_cr, method='ward', metric='euclidean')
# CAH et affichage du dendrogramme
plt.title("CAH")
dendrogram(Z_subset, labels=fromage_subset.index, orientation='left', color_threshold=7)
plt.show()
# Groupes
groupes_subset_cah = fcluster(Z_subset, t=7, criterion='distance')
# Sélectionner seulement les colonnes numériques
fromage_df = fromage.select_dtypes(include=['number'])
# Effectuer le clustering avec KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(fromage_df)
# Réduire la dimensionnalité avec PCA
pca = PCA(n_components=2)
fromage_pca = pca.fit_transform(fromage_df)
# Tracer les clusters avec les noms des fromages
plt.figure(figsize=(10, 6))
for i in range(len(fromage_df)):
    plt.scatter(fromage_pca[i, 0], fromage_pca[i, 1], c='blue')  # Tous les fromages sont affichés en bleu
    plt.text(fromage_pca[i, 0], fromage_pca[i, 1], fromage_df.index[i], fontsize=8)
# Afficher les centroids des clusters
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red', s=100, label='Centroids')
plt.title('Clusters de fromages avec noms')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.legend()
plt.grid(True)
plt.show()
