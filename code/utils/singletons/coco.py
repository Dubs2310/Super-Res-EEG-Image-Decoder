import os
import mne
import json
import clip
import yaml
import torch
import random
import requests
import open_clip
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from urllib.parse import urlparse
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict

def load_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        return {}

CONFIG = load_config()
DEFAULT_METADATA_PATH = CONFIG.get('coco', {}).get('metadata_path', "/workspace/eeg-image-decoding/data/all-joined-1/coco/train17-images-metadata.json")
DEFAULT_IMAGE_DIR = CONFIG.get('coco', {}).get('image_dir', "/workspace/eeg-image-decoding/data/all-joined-1/coco/images")
DEFAULT_FEATURES_DIR = CONFIG.get('coco', {}).get('features_dir', "/workspace/eeg-image-decoding/data/all-joined-1/coco/features")
DEFAULT_OUTPUT_DIR = CONFIG.get('coco', {}).get('dataframe_dir', "/workspace/eeg-image-decoding/data/all-joined-1/coco/dataframes")
DEFAULT_EEG_PATH = CONFIG.get('eeg', {}).get('preprocessed_dir', "/workspace/eeg-image-decoding/data/all-joined-1/eeg/preprocessed")


class COCO:
    _instance = None
    _initialized = False
    
    def __new__(cls, file_path=None):
        if cls._instance is None:
            cls._instance = super(COCO, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, file_path=None):
        if not self._initialized:
            if file_path is None:
                file_path = DEFAULT_METADATA_PATH
            self.file_path = file_path
            self._load_dataset()
            self._extract_categories()
            self._setup_openclip_model()
            self._initialized = True
    
    def _load_dataset(self):
        with open(self.file_path, 'r') as file:
            self.coco_dataset = json.load(file)
    
    def __getitem__(self, idx):
        return self.coco_dataset[idx]
    
    def _extract_categories(self):
        all_fine_categories = set()
        all_super_categories = set()
        self.category_distributions = dict()
        self.supercategory_to_fine = defaultdict(set)
        for coco_metadata in self.coco_dataset:
            for coco_categories in coco_metadata['categories']:
                id, cat, sup_cat = coco_categories['category_id'], coco_categories['category_name'], coco_categories['supercategory_name']
                all_fine_categories.add(cat)
                all_super_categories.add(sup_cat)
                self.supercategory_to_fine[sup_cat].add(cat)
                if not (id, sup_cat, cat) in self.category_distributions:
                    self.category_distributions[(id, sup_cat, cat)] = 1
                else:
                    self.category_distributions[(id, sup_cat, cat)] += 1
        self.all_fine_categories = list(all_fine_categories)
        self.all_super_categories = list(all_super_categories)
        for sup_cat in self.supercategory_to_fine:
            self.supercategory_to_fine[sup_cat] = list(self.supercategory_to_fine[sup_cat])
    
    def _setup_openclip_model(self):
        model_type = 'ViT-H-14'
        pretrained = 'laion2b_s32b_b79k'
        precision = 'fp32'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vlmodel, self.preprocess_train, self.feature_extractor = open_clip.create_model_and_transforms(model_type, pretrained=pretrained, precision=precision, device=self.device)
    
    def get_fine_category_labels(self):
        return sorted(self.all_fine_categories)
    
    def get_supercategory_labels(self):
        return sorted(self.all_super_categories)
    
    def get_fine_category_mask_from_active_supercategories(self, supercategories_list):
        supercategory_labels = sorted([sc for sc in self.all_super_categories])
        fine_category_labels = sorted(self.all_fine_categories)
        fine_category_labels = [fc for fc in fine_category_labels]
        mask_df = pd.DataFrame(columns=["img_id"] + supercategory_labels + fine_category_labels)
        mask_df = mask_df.loc[:, ~mask_df.columns.duplicated()]
        mask_df.loc[0] = 0
        mask_df.loc[0, "img_id"] = -1
        for supercategory in supercategories_list:
            if supercategory in self.all_super_categories:
                mask_df.loc[0, supercategory] = 1
                related_fine_cats = self.supercategory_to_fine[supercategory]
                for cat in related_fine_cats:
                    mask_df.loc[0, cat] = 1
        return mask_df
    
    def _download_image(self, img_url, save_directory, filename=None):
        os.makedirs(save_directory, exist_ok=True)
        if filename is None:
            filename = os.path.basename(urlparse(img_url).path)
        save_path = os.path.join(save_directory, filename)
        try:
            response = requests.get(img_url, stream=True)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
            return save_path
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {img_url}: {e}")
            return None
    
    def download_images(self, save_directory):
        for i, img_meta in enumerate(self.coco_dataset):
            filename = f"{i}.jpg"
            save_path = os.path.join(save_directory, filename)
            if os.path.exists(save_path):
                print(f"Image {filename} already exists, skipping...")
                continue
            split = img_meta['cocoSplit']
            id = img_meta['cocoId']
            img_url = f"http://images.cocodataset.org/{split}/{'0' * (12 - len(id))}{id}.jpg"
            print(f"Downloading from {img_url} ...")
            self._download_image(img_url, save_directory, filename)
    
    def _create_categorical_dataframe(self):
        supercategory_labels = sorted([sc for sc in self.all_super_categories])
        fine_category_labels = sorted(self.all_fine_categories)
        fine_category_labels = [fc for fc in fine_category_labels]
        img_cat_df = pd.DataFrame(columns=["img_id"] + supercategory_labels + fine_category_labels)
        img_cat_df = img_cat_df.loc[:, ~img_cat_df.columns.duplicated()]
        for img_id, img_meta in enumerate(self.coco_dataset):
            img_cat_df.loc[img_id] = 0
            img_cat_df.loc[img_id, "img_id"] = img_id
            for img_categories in img_meta['categories']:
                cat, sup_cat = img_categories['category_name'], img_categories['supercategory_name']
                img_cat_df.loc[img_id, sup_cat] = 1
                img_cat_df.loc[img_id, cat] = 1
        return img_cat_df
    
    def generate_eeg_image_pairs_dataframe(self, eeg_path, output_path=None):
        img_cat_df = self._create_categorical_dataframe()
        files = os.listdir(eeg_path)
        file_paths = [os.path.join(eeg_path, f) for f in files]
        all_dfs = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            raw = mne.io.read_raw_fif(file_path, preload=True)
            img_evts = mne.find_events(raw)
            subject = int(filename[4:6])
            session = int(filename[14:15])
            img_evts_data = [[subject, session, i, int(img[-1] - 1)] for i, img in enumerate(img_evts)]
            eeg_df = pd.DataFrame(columns=['subject', 'session', 'epoch_idx', 'img_id'], data=img_evts_data)
            df = pd.merge(eeg_df, img_cat_df, on='img_id', how='left')
            all_dfs.append(df)
        final_df = pd.concat(all_dfs, ignore_index=True)
        if output_path:
            final_df.to_csv(output_path, index=False)
            print(f"Saved combined dataframe with shape {final_df.shape} to {output_path}")
        return final_df
    
    def cluster_images_with_clip(self, image_dir, n_clusters=20, test_size=160, seed=42):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        def get_embedding(img_path):
            try:
                image = self.preprocess_train(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    return self.vlmodel.encode_image(image).squeeze().cpu().numpy()
            except Exception as e:
                print(f"Error with {img_path}: {e}")
                return None
        
        print("Extracting embeddings...")
        embeddings = [get_embedding(path) for path in image_paths]
        valid = [e is not None for e in embeddings]
        embeddings = np.array([e for e in embeddings if e is not None])
        image_paths = [p for p, v in zip(image_paths, valid) if v]
        
        print("Clustering embeddings...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        cluster_ids = kmeans.fit_predict(embeddings)
        
        train_images, test_images = self._split_by_cluster_proportional(image_paths, cluster_ids, test_size, seed)
        
        self.cluster_embeddings = embeddings
        self.cluster_ids = cluster_ids
        self.cluster_image_paths = image_paths
        self.train_images = train_images
        self.test_images = test_images
        
        return train_images, test_images, cluster_ids
    
    def _split_by_cluster_proportional(self, image_paths, cluster_ids, test_size=160, seed=42):
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
    
    def visualize_clusters(self, n_clusters=20, seed=42):
        if not hasattr(self, 'cluster_embeddings'):
            print("No cluster data found. Run cluster_images_with_clip first.")
            return
        
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
        embedding_2d = tsne.fit_transform(self.cluster_embeddings)

        plt.figure(figsize=(10, 8))
        palette = sns.color_palette("hls", n_clusters)
        sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=self.cluster_ids, palette=palette, s=40, edgecolor='k', linewidth=0.2, alpha=0.8)
        plt.title("t-SNE Visualization of Image Clusters")
        plt.tight_layout()
        plt.show()
    
    def setup_dataset(self, eeg_path=None, image_dir=None, features_dir=None, dataframe_dir=None, n_clusters=20, test_size=160, seed=42):
        eeg_path = eeg_path or DEFAULT_EEG_PATH
        image_dir = image_dir or DEFAULT_IMAGE_DIR
        features_dir = features_dir or DEFAULT_FEATURES_DIR
        dataframe_dir = dataframe_dir or DEFAULT_OUTPUT_DIR
        
        os.makedirs(dataframe_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        
        print("Step 1: Checking and downloading images...")
        if not self._check_images_exist(image_dir):
            print("Images not found, downloading...")
            self.download_images(image_dir)
        else:
            print("Images already exist, skipping download.")
        
        print("Step 2: Generating EEG-image dataframe and train/test split...")
        global_df = self.generate_eeg_image_pairs_dataframe(eeg_path)
        train_images, test_images, cluster_ids = self.cluster_images_with_clip(image_dir, n_clusters, test_size, seed)
        
        train_img_ids = [int(os.path.basename(path).split('.')[0]) for path in train_images]
        test_img_ids = [int(os.path.basename(path).split('.')[0]) for path in test_images]
        
        print("Step 3: Creating train/test dataframes...")
        train_df = global_df[global_df['img_id'].isin(train_img_ids)].reset_index(drop=True)
        test_df = global_df[global_df['img_id'].isin(test_img_ids)].reset_index(drop=True)
        
        global_df_path = os.path.join(dataframe_dir, 'global_eeg_image_df.csv')
        train_df_path = os.path.join(dataframe_dir, 'train_eeg_image_df.csv')
        test_df_path = os.path.join(dataframe_dir, 'test_eeg_image_df.csv')
        
        global_df.to_csv(global_df_path, index=False)
        train_df.to_csv(train_df_path, index=False)
        test_df.to_csv(test_df_path, index=False)
        
        print(f"Saved dataframes:")
        print(f"  Global: {global_df_path} ({len(global_df)} records)")
        print(f"  Train: {train_df_path} ({len(train_df)} records)")
        print(f"  Test: {test_df_path} ({len(test_df)} records)")
        
        print("Step 4: Generating image and text features...")
        train_features = self.get_image_text_features(train_df, image_dir, features_dir, 'train_features.pt')
        test_features = self.get_image_text_features(test_df, image_dir, features_dir, 'test_features.pt')
        
        print("Dataset setup complete!")
        
        return {
            'global_df': global_df,
            'train_df': train_df,
            'test_df': test_df,
            'train_features': train_features,
            'test_features': test_features
        }
    
    def _check_images_exist(self, image_dir):
        if not os.path.exists(image_dir):
            return False
        expected_count = len(self.coco_dataset)
        actual_count = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        return actual_count >= expected_count
    
    def get_global_eeg_image_df(self, dataframe_dir=None):
        dataframe_dir = dataframe_dir or DEFAULT_OUTPUT_DIR
        global_df_path = os.path.join(dataframe_dir, 'global_eeg_image_df.csv')
        if os.path.exists(global_df_path):
            return pd.read_csv(global_df_path)
        else:
            print(f"Global dataframe not found at {global_df_path}. Run setup_dataset() first.")
            return None
    
    def get_train_set(self, dataframe_dir=None, features_dir=None):
        dataframe_dir = dataframe_dir or DEFAULT_OUTPUT_DIR
        features_dir = features_dir or DEFAULT_FEATURES_DIR
        train_df_path = os.path.join(dataframe_dir, 'train_eeg_image_df.csv')
        train_features_path = os.path.join(features_dir, 'train_features.pt')
        
        if not os.path.exists(train_df_path):
            print(f"Train dataframe not found at {train_df_path}. Run setup_dataset() first.")
            return None, None, None
        
        if not os.path.exists(train_features_path):
            print(f"Train features not found at {train_features_path}. Run setup_dataset() first.")
            return None, None, None
        
        train_df = pd.read_csv(train_df_path)
        train_features = torch.load(train_features_path)
        
        return train_df, train_features['image_features'], train_features['text_features'], train_features['img_id_to_indices']
    
    def get_test_set(self, dataframe_dir=None, features_dir=None):
        dataframe_dir = dataframe_dir or DEFAULT_OUTPUT_DIR
        features_dir = features_dir or DEFAULT_FEATURES_DIR
        test_df_path = os.path.join(dataframe_dir, 'test_eeg_image_df.csv')
        test_features_path = os.path.join(features_dir, 'test_features.pt')
        
        if not os.path.exists(test_df_path):
            print(f"Test dataframe not found at {test_df_path}. Run setup_dataset() first.")
            return None, None, None
        
        if not os.path.exists(test_features_path):
            print(f"Test features not found at {test_features_path}. Run setup_dataset() first.")
            return None, None, None
        
        test_df = pd.read_csv(test_df_path)
        test_features = torch.load(test_features_path)
        
        return test_df, test_features['image_features'], test_features['text_features'], test_features['img_id_to_indices']
    
    def preview_cluster_samples(self, images_per_cluster=5, thumb_size=(100, 100)):
        if not hasattr(self, 'cluster_image_paths'):
            print("No cluster data found. Run cluster_images_with_clip first.")
            return
        
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

        sampled = sample_cluster_images(self.cluster_image_paths, self.cluster_ids, n_samples=images_per_cluster)
        plot_cluster_samples(sampled, images_per_cluster=images_per_cluster, thumb_size=thumb_size)
    
    def _create_descriptive_caption(self, super_categories, fine_categories):
        # Group fine categories by super category
        super_to_fine = defaultdict(list)
        for fine_cat in fine_categories:
            # Find which super category this fine category belongs to
            for img_cat in self.coco_dataset:
                for cat_info in img_cat['categories']:
                    if cat_info['category_name'] == fine_cat:
                        super_to_fine[cat_info['supercategory_name']].append(fine_cat)
                        break
                if fine_cat in super_to_fine[cat_info['supercategory_name']]:
                    break
        
        # Remove duplicates and sort
        for super_cat in super_to_fine:
            super_to_fine[super_cat] = sorted(list(set(super_to_fine[super_cat])))
        
        description_parts = []
        
        for super_cat in sorted(super_categories):
            fine_cats_for_super = super_to_fine.get(super_cat, [])
            
            # Handle different super categories with appropriate grammar
            if super_cat == 'person':
                if len(fine_cats_for_super) > 0 and fine_cats_for_super != ['person']:
                    # If fine categories are different from just 'person'
                    fine_text = ', '.join(fine_cats_for_super)
                    description_parts.append(f"person/people (specifically {fine_text})")
                else:
                    description_parts.append("person/people")
            else:
                # For other categories, use singular/plural appropriately
                if len(fine_cats_for_super) == 0:
                    description_parts.append(super_cat)
                elif len(fine_cats_for_super) == 1:
                    if fine_cats_for_super[0] == super_cat:
                        description_parts.append(super_cat)
                    else:
                        description_parts.append(f"{super_cat} (specifically {fine_cats_for_super[0]})")
                else:
                    fine_text = ', '.join(fine_cats_for_super)
                    description_parts.append(f"{super_cat}(s) (specifically {fine_text})")
        
        if description_parts:
            if len(description_parts) == 1:
                return f"This image contains {description_parts[0]}."
            else:
                return f"This image contains {', '.join(description_parts[:-1])} and {description_parts[-1]}."
        else:
            return ""

    def get_image_text_features(self, df, img_dir, features_dir, features_file):
        unique_img_ids = df['img_id'].unique()
        
        img_id_to_captions = {}
        img_id_to_images = {}
        all_captions = []
        all_images = []
        img_id_mapping = []
        
        for img_id in unique_img_ids:
            img_captions = self.coco_dataset[img_id]['captions']
            img_categories = self.coco_dataset[img_id]['categories']
            
            # Extract super and fine categories
            super_categories = list(set([cat['supercategory_name'] for cat in img_categories]))
            fine_categories = list(set([cat['category_name'] for cat in img_categories]))
            
            # Create descriptive caption
            descriptive_caption = self._create_descriptive_caption(super_categories, fine_categories)
            
            # Use descriptive caption for all instances of this image
            augmented_captions = [descriptive_caption] * len(img_captions)
            
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            
            start_idx = len(all_captions)
            
            all_captions.extend(augmented_captions)
            all_images.extend([img_path] * len(augmented_captions))
            
            img_id_to_captions[img_id] = list(range(start_idx, start_idx + len(augmented_captions)))
            img_id_to_images[img_id] = list(range(start_idx, start_idx + len(augmented_captions)))

        features_file_path = os.path.join(features_dir, features_file)
        if os.path.exists(features_file_path):
            print(f"Loading existing features from {features_file_path}")
            saved_features = torch.load(features_file_path)
            text_features = saved_features['text_features']
            image_features = saved_features['image_features']
            img_id_to_indices = saved_features['img_id_to_indices']
        else:
            print(f"Computing new features and saving to {features_file_path}")
            text_features = self._encode_text_captions(all_captions)
            image_features = self._encode_images(all_images)
            
            img_id_to_indices = {}
            for img_id in unique_img_ids:
                indices = img_id_to_captions[img_id]
                img_id_to_indices[int(img_id)] = indices
            
            os.makedirs(features_dir, exist_ok=True)
            torch.save({ 
                'text_features': text_features.cpu(), 
                'image_features': image_features.cpu(),
                'img_id_to_indices': img_id_to_indices
            }, features_file_path)

        return {
            'text_features': text_features,
            'image_features': image_features,
            'img_id_to_indices': img_id_to_indices
        }

    def _encode_text_captions(self, texts):
        batch_size = 50
        text_features_list = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            text_inputs = torch.cat([clip.tokenize(t) for t in batch_texts]).to(self.device)
            
            with torch.no_grad():
                batch_text_features = self.vlmodel.encode_text(text_inputs)
                batch_text_features = F.normalize(batch_text_features, dim=-1)
            
            text_features_list.append(batch_text_features.cpu())
            
            del text_inputs, batch_text_features
            torch.cuda.empty_cache()
        
        return torch.cat(text_features_list, dim=0)
    
    def _encode_images(self, images):
        batch_size = 10
        image_features_list = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            try:
                image_inputs = torch.stack([
                    self.preprocess_train(Image.open(img).convert("RGB")) 
                    for img in batch_images
                ]).to(self.device)
                
                with torch.no_grad():
                    batch_image_features = self.vlmodel.encode_image(image_inputs)
                    batch_image_features = F.normalize(batch_image_features, dim=-1)
                
                image_features_list.append(batch_image_features.cpu())
                
                del image_inputs, batch_image_features
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing image batch {i}: {e}")
                dummy_features = torch.zeros(len(batch_images), 1024)
                image_features_list.append(dummy_features)
        
        return torch.cat(image_features_list, dim=0)

    def plot_categorical_distributions(self):
        super_cat_counts = defaultdict(int)
        cat_data = []
        
        for (id_num, super_cat, cat), count in self.category_distributions.items():
            super_cat_counts[super_cat] += count
            cat_data.append({'super_category': super_cat, 'category': cat, 'count': count})
        
        df = pd.DataFrame(cat_data)
        
        super_cats = list(super_cat_counts.keys())
        super_counts = list(super_cat_counts.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(super_cats, super_counts, color='skyblue', alpha=0.7)
        plt.title('Distribution by Super Categories', fontsize=14, fontweight='bold')
        plt.xlabel('Super Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        df_sorted = df.sort_values('count', ascending=True)
        categories = df_sorted['category'].tolist()
        counts = df_sorted['count'].tolist()
        
        super_cat_colors = {
            'person': 'red', 'vehicle': 'blue', 'outdoor': 'green',
            'animal': 'orange', 'accessory': 'purple', 'sports': 'brown',
            'kitchen': 'pink', 'food': 'gray', 'furniture': 'olive',
            'electronic': 'cyan', 'appliance': 'magenta', 'indoor': 'yellow'
        }
        
        colors = [super_cat_colors.get(df_sorted.iloc[i]['super_category'], 'black')
                  for i in range(len(categories))]
        
        plt.figure(figsize=(10, 25))
        bars = plt.barh(range(len(categories)), counts, color=colors, alpha=0.7)
        plt.title('Distribution by Individual Categories (Sorted by Count)', fontsize=16, fontweight='bold')
        plt.ylabel('Categories', fontsize=14)
        plt.xlabel('Count', fontsize=14)
        plt.yticks(range(len(categories)), categories, fontsize=8)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2.,
                     f'{int(width)}', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 35))
        
        y_pos = 0
        y_ticks = []
        y_labels = []
        
        for super_cat in super_cat_counts.keys():
            super_cat_data = df[df['super_category'] == super_cat].sort_values('count', ascending=True)
            cat_names = super_cat_data['category'].tolist()
            cat_counts = super_cat_data['count'].tolist()
            
            y_positions = range(y_pos, y_pos + len(cat_names))
            bars = plt.barh(y_positions, cat_counts,
                            color=super_cat_colors.get(super_cat, 'black'),
                            alpha=0.7, label=super_cat)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 1, bar.get_y() + bar.get_height()/2.,
                         f'{int(width)}', ha='left', va='center', fontsize=8)
            
            y_ticks.extend(y_positions)
            y_labels.extend(cat_names)
            
            y_pos += len(cat_names) + 1
        
        plt.title('All Categories Grouped by Super Category', fontsize=18, fontweight='bold')
        plt.ylabel('Categories', fontsize=14)
        plt.xlabel('Count', fontsize=14)
        plt.yticks(y_ticks, y_labels, fontsize=9)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()