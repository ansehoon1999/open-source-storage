# Import list 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs

# Random Seed Set.
SEED = 30

# Load Data set.
data = pd.read_csv("segmentation data.csv")

# Make test Data X.
X = data[['Age','Income']]

# Make PAM clustering model.
pam = KMedoids(4,method='pam',random_state=SEED)
pam.fit(X)

# check Lables of each point.
labels = pam.labels_

# Draw clustering result. (Medoids are represented in black.) 
unique_labels = set(labels)

colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]

for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k
    xy = X[class_member_mask]

    plt.plot(
        xy['Age'],
        xy['Income'],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
    
plt.plot(
    pam.cluster_centers_[:,0],
    pam.cluster_centers_[:,1],
    "o",
    markerfacecolor="black",
    markeredgecolor="k",
    markersize=6,
)

plt.title("KMedoids clustering. Medoids are represented in black.")
