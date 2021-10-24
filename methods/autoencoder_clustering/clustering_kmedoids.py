import torch
import os
import pandas as pd
from torchvision.models import mobilenet_v2
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from ae import AE
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.metrics import accuracy_score

ckpt_path = "auto_encoder_train_500epochs.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_size = 128
nr_annotators = 3
classes_names = ['pyriform', 'amorphous', 'normal', 'tapered']
BASE_DATA_PATH = "/home/icrto/Documents/MICCAIHackathon/data"
df = pd.read_csv("/home/icrto/Documents/MICCAIHackathon/file.csv")
train_df = df.loc[(df['split'] == 0) | (df['split'] == 2)]
test_df = df.loc[(df['split'] == 1)]

encoder = AE()
encoder = torch.load(ckpt_path)
encoder.to(device)
encoder.eval()

transforms = Compose([Resize((img_size, img_size)), ToTensor()])

extracted_features = []
print("Extracting features...")
with torch.no_grad():
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img = Image.open(os.path.join(BASE_DATA_PATH, row['img']) + '.BMP')
        img = transforms(img)
        img = img.to(device)
        img = img[None, :, :, :]
        feats = encoder.get_embeds(img)
        feats = torch.flatten(feats)
        feats = feats.cpu().detach().numpy()
        extracted_features.append(feats)

train_df['features'] = extracted_features
print("Clustering...")
X = np.array(extracted_features)
kmed = KMedoids(n_clusters=4, random_state=42, init="heuristic", method="pam", metric="cosine").fit(X)
cluster_output = kmed.predict(X)
train_df['clustering'] = cluster_output
cluster_ids = np.unique(cluster_output)

medoids_embeds = kmed.cluster_centers_
medoids_indices = kmed.medoid_indices_
medoids_labels = train_df.iloc[medoids_indices]["label"].values

print("Gathering clustering results...")
final_stats_class = {}
final_stats_dist = {}
final_stats_medoids = {}

for cid in cluster_ids:
    print(f"\nCluster ID {cid}:")
    cluster_df = train_df[train_df['clustering'] == cid]
    print(f" Total number of samples in cluster: {len(cluster_df)}")
    cluster_stats_idx = list(range(4))
    cluster_lens = {}
    cluster_dists = {}
    for i in range(4): # for each class
        class_df = cluster_df[cluster_df['label'] == i]
        class_len = len(class_df)
        cluster_lens[i] = class_len

        # compute mean of distances to centroids
        cluster_dists[i] = 0
        medoid_embeds = medoids_embeds[cid]
        if class_len > 0:
            for _, x in class_df.iterrows():
                cluster_dists[i] += cosine_distances(x['features'].reshape(1, -1), medoid_embeds.reshape(1, -1))[0][0]
            cluster_dists[i] /= class_len
            print(f" Number of annotator given label {i} ({classes_names[i]}): {class_len} / {np.round(cluster_dists[i], 4)}")
        else:
            cluster_dists[i] = 10000000
            print(f" Number of annotator given label {i} ({classes_names[i]}): {class_len} / NA")

    majority_class = max(cluster_lens, key=cluster_lens.get)
    majority_dist = min(cluster_dists, key=cluster_dists.get)
    majority_medoid = medoids_labels[cid]

    print(f" Voting 'label' of the cluster: {classes_names[majority_class]} / {classes_names[majority_dist]} / {classes_names[majority_medoid]}")
    final_stats_class[cid] = majority_class
    final_stats_dist[cid] = majority_dist
    final_stats_medoids[cid] = majority_medoid


# X_embedded_tsne = TSNE(n_components=2).fit_transform(X)

# plt.figure()
# plt.scatter(X_embedded_tsne[:,0], X_embedded_tsne[:,1],c=cluster_output)
# plt.show()

print("\nComputing annotator consistency...")
for i in range(1, nr_annotators+1):
    ann_test_df = test_df[test_df['annotator'] == i]
    preds_annotator = []
    preds_clustering_class = [] 
    preds_clustering_dist = []
    preds_clustering_medoids = []
    for _, row in ann_test_df.iterrows():
        img = Image.open(os.path.join(BASE_DATA_PATH, row['img']) + '.BMP')
        img = transforms(img)
        img = img.to(device)
        img = img[None, :, :, :]
        feats = encoder.get_embeds(img)
        feats = torch.flatten(feats)
        feats = feats.cpu().detach().numpy()
        feats = feats.reshape(1, -1)

        kmed_cluster_id = kmed.predict(feats)
        kmed_label_class = final_stats_class[kmed_cluster_id[0]]
        kmed_label_dist = final_stats_dist[kmed_cluster_id[0]]
        kmed_label_medoids = final_stats_medoids[kmed_cluster_id[0]]

        preds_clustering_class.append(kmed_label_class)
        preds_clustering_dist.append(kmed_label_dist)
        preds_clustering_medoids.append(kmed_label_medoids)
        preds_annotator.append(row['label'])

    ck_class = cohen_kappa_score(preds_annotator, preds_clustering_class, labels=[0, 1, 2, 3])
    ck_dist = cohen_kappa_score(preds_annotator, preds_clustering_dist, labels=[0, 1, 2, 3])
    ck_medoids = cohen_kappa_score(preds_annotator, preds_clustering_medoids, labels=[0, 1, 2, 3])

    acc_class = accuracy_score(preds_clustering_class, preds_annotator)
    acc_dist = accuracy_score(preds_clustering_dist, preds_annotator)
    acc_medoids = accuracy_score(preds_clustering_medoids, preds_annotator)


    print(f"Annotator {i}:")
    print(f"Cohen's kappa: {np.round(ck_class, 4)} / {np.round(ck_dist,4)} / {np.round(ck_medoids,4)}")
    print(f"% failures: {np.round(1-acc_class, 4)} / {np.round(1-acc_dist,4)} / {np.round(1-acc_medoids,4)}")


X_embedded_tsne = PCA(n_components=2).fit_transform(X)

plt.figure(1)
for cid in cluster_ids:
    idxs = np.where(cluster_output == cid)
    x_cid = X_embedded_tsne[idxs]
    annotators = train_df.iloc[idxs]["annotator"].values
    plt.scatter(x_cid[:,0], x_cid[:,1]-0.03, label="cluster id " + str(cid))
    for i in range(len(annotators)):
        plt.text(x_cid[i,0], x_cid[i,1], annotators[i], fontsize=9)
plt.legend()
plt.title("Clusters with respective annotators")

plt.figure(2)
for cid in cluster_ids:
    idxs = np.where(cluster_output == cid)
    x_cid = X_embedded_tsne[idxs]
    labels = train_df.iloc[idxs]["label"].values
    plt.scatter(x_cid[:,0], x_cid[:,1]-0.03, label="cluster id " + str(cid))
    for i in range(len(labels)):
        plt.text(x_cid[i,0], x_cid[i,1], labels[i], fontsize=9)
plt.legend()
plt.title("Clusters with respective class labels")
plt.show()
