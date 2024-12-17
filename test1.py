#chargement data
import  pandas as pd
namefile= 'fromage.txt'
data= pd.read_csv(namefile,sep=r'\s+')
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
print(data.head(29))

# stat descriptives
import  pandas as pd
namefile= 'fromage.txt'
data= pd.read_csv(namefile,sep=r'\s+')
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', None)
description = data.describe()
print(description)
description = data.describe()
print(description)

# histogrammes
import matplotlib.pyplot as plt
import pandas as pd
namefile= 'fromage.txt'
data= pd.read_csv(namefile,sep=r'\s+')
data.hist(figsize=(9, 8))
plt.show()



# plots
import matplotlib.pyplot as plt
import pandas as pd
namefile= 'fromage.txt'
data= pd.read_csv(namefile,sep=r'\s+')
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(10,6))
plt.show()


# valeur manquantes

import pandas as pd
namefile= 'fromage.txt'
data= pd.read_csv(namefile,sep=r'\s+')
print(data.isnull().sum())



# corrélation
import pandas as pd
namefile = 'fromage.txt'
data = pd.read_csv(namefile, sep=r'\s+', index_col=0)
correlations = data.corr(method='pearson')
print(correlations)




 # CAh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# Charger les données depuis le fichier fromage.txt
filename = 'fromage.txt'
data = pd.read_csv(filename, sep=r'\s+', index_col=0)

# calculer la matrice de laison pour la cah
z = linkage(data, method='ward')

# Afficher le dendrogramme
plt.figure(figsize=(12, 8))
dendrogram (z,labels=data.index,leaf_rotation=90)
plt.title('Dendrogramme de la classification ascendante hiérarchique')
plt.xlabel('Fromages')
plt.ylabel('Distance')
plt.show()

