import os
import clip
import torch
import random
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict

# === CONFIG ===
SEED = 42
N_CLUSTERS = 20
TEST_SIZE = 160  # out of 960
IMAGES_PER_CLUSTER_PREVIEW = 5
IMAGE_DIR = "S:\\PolySecLabProjects\\eeg-image-decoding\\data\\all-joined-1\\coco\\images"

# === DEVICE SETUP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# === LOAD IMAGE PATHS ===
image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]

# === EMBEDDING FUNCTION ===
def get_embedding(img_path):
    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_image(image).squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error with {img_path}: {e}")
        return None

print("Extracting embeddings...")
embeddings = [get_embedding(path) for path in image_paths]
valid = [e is not None for e in embeddings]
embeddings = np.array([e for e in embeddings if e is not None])
image_paths = [p for p, v in zip(image_paths, valid) if v]

# === CLUSTERING ===
print("Clustering embeddings...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED)
cluster_ids = kmeans.fit_predict(embeddings)

# === T-SNE VISUALIZATION ===
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
embedding_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
palette = sns.color_palette("hls", N_CLUSTERS)
sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=cluster_ids, palette=palette, s=40, edgecolor='k', linewidth=0.2, alpha=0.8)
plt.title("t-SNE Visualization of Image Clusters")
plt.tight_layout()
plt.show()

# === TRAIN/TEST SPLIT BY CLUSTER PROPORTION ===
def split_by_cluster_proportional(image_paths, cluster_ids, test_size=160, seed=42):
    random.seed(seed)
    total_images = len(image_paths)
    cluster_to_indices = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(idx)

    cluster_test_allocs = {
        cid: round(len(idxs) / total_images * test_size)
        for cid, idxs in cluster_to_indices.items()
    }

    test_indices = set()
    for cid, n in cluster_test_allocs.items():
        samples = random.sample(cluster_to_indices[cid], min(n, len(cluster_to_indices[cid])))
        test_indices.update(samples)

    all_indices = set(range(total_images))
    train_indices = sorted(all_indices - test_indices)
    test_indices = sorted(test_indices)

    train_images = [image_paths[i] for i in train_indices]
    test_images = [image_paths[i] for i in test_indices]

    return train_images, test_images

train_images, test_images = split_by_cluster_proportional(image_paths, cluster_ids, test_size=TEST_SIZE, seed=SEED)
print(f"Train images: {len(train_images)}, Test images: {len(test_images)}")

# === CLUSTER THUMBNAIL PREVIEW ===
def sample_cluster_images(image_paths, cluster_ids, n_samples=5):
    cluster_to_images = defaultdict(list)
    for path, cid in zip(image_paths, cluster_ids):
        cluster_to_images[cid].append(path)
    sampled = {cid: random.sample(paths, min(n_samples, len(paths))) for cid, paths in cluster_to_images.items()}
    return sampled

def plot_cluster_samples(sampled_images, images_per_cluster=5, thumb_size=(100, 100)):
    n_clusters = len(sampled_images)
    fig, axes = plt.subplots(n_clusters, images_per_cluster, figsize=(images_per_cluster * 2, n_clusters * 2))

    for row, (cid, paths) in enumerate(sorted(sampled_images.items())):
        for col in range(images_per_cluster):
            ax = axes[row][col]
            ax.axis("off")
            try:
                img = Image.open(paths[col]).resize(thumb_size)
                ax.imshow(img)
            except Exception as e:
                ax.set_facecolor("lightgray")
                print(f"Error loading image: {e}")
        axes[row][0].set_ylabel(f"Cluster {cid}", fontsize=10)

    plt.tight_layout()
    plt.show()

sampled = sample_cluster_images(image_paths, cluster_ids, n_samples=IMAGES_PER_CLUSTER_PREVIEW)
plot_cluster_samples(sampled, images_per_cluster=IMAGES_PER_CLUSTER_PREVIEW)