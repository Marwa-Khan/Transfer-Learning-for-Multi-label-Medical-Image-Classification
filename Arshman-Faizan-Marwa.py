#!/usr/bin/env python
# coding: utf-8

# ##  Loading the libraries and dataframes

# In[1]:


#!pip install timm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18 
import timm
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.amp import autocast, GradScaler

# --- GLOBAL CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = '/kaggle/input/final-project-deep-learning-fall-2025/final_project_resources'
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
IMG_SIZE = 256
DISEASE_COLUMNS = ['D', 'G', 'A']
TEAM_ID = "Arshman-Faizan-Marwa" # Updated Kaggle Team Name 

# --- LOAD DATAFRAMES ---
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(BASE_DIR, 'val.csv'))
offsite_test_df = pd.read_csv(os.path.join(BASE_DIR, 'offsite_test.csv'))

print(f"Setup complete. Training on: {DEVICE}")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Offsite Test samples: {len(offsite_test_df)}")


# ##  Implement Image Dataset and DataLoaders

# In[2]:


class ODIRDataset(Dataset):
    def __init__(self, df, image_dir, mode='train'):
        self.df = df
        self.image_dir = image_dir
        self.mode = mode
        
        if mode == 'train':
            self.image_folder = os.path.join(image_dir, 'train')
        elif mode == 'val':
            self.image_folder = os.path.join(image_dir, 'val')
        elif mode in ['test', 'offsite_test']:
            self.image_folder = os.path.join(image_dir, 'offsite_test') 
        elif mode == 'test_onsite':
            self.image_folder = os.path.join(image_dir, 'onsite_test')
        
        self.transform = self._get_transforms(mode)

    def _get_transforms(self, mode):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, row['id'])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        if self.mode != 'test_onsite': 
            labels = torch.tensor(row[DISEASE_COLUMNS].values.astype(float), dtype=torch.float32)
            return image, labels
        return image, row['id']

# Initialize Loaders
train_loader = DataLoader(ODIRDataset(train_df, IMAGE_DIR, 'train'), batch_size=32, shuffle=True)
val_loader = DataLoader(ODIRDataset(val_df, IMAGE_DIR, 'val'), batch_size=32, shuffle=False)
offsite_loader = DataLoader(ODIRDataset(offsite_test_df, IMAGE_DIR, 'test'), batch_size=32, shuffle=False)


# ## Data Exploration and Disease Distribution Visualization
# 

# In[3]:


# Columns representing the three major diseases
DISEASE_COLUMNS = ['D', 'G', 'A']

# Calculate the count of each disease in the training set
disease_counts = train_df[DISEASE_COLUMNS].sum()

# --- Plotting the Distribution  ---
plt.figure(figsize=(8, 6))
bars = plt.bar(disease_counts.index, disease_counts.values, color=['green', 'blue', 'red'])

# Add the counts on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', va='bottom', fontsize=12)

plt.title('Distribution of Diseases in Training Dataset')
plt.xlabel('Disease Type')
plt.ylabel('Number of Images')
plt.ylim(0, 550) # Set limit based on the document's visualization
plt.grid(axis='y', linestyle='--')

# Save the plot for the report 
CHART_PATH = 'disease_distribution_train.png'
plt.savefig(CHART_PATH)
plt.show()

print(f"Disease distribution chart saved to: {CHART_PATH}")
# Output will confirm the imbalance: D: 517, G: 163, A: 142 [cite: 50, 56, 57]


# ## VAE Image Generation

# In[4]:


# =========================
# Create writable VAE training image directory
# =========================

VAE_TRAIN_DIR = "/kaggle/working/images/train"
os.makedirs(VAE_TRAIN_DIR, exist_ok=True)

print("Writable VAE train directory:", VAE_TRAIN_DIR)


# In[5]:


# =========================
# Copy original training images to working directory
# =========================

import shutil

ORIG_TRAIN_DIR = os.path.join(IMAGE_DIR, "train")

for fname in os.listdir(ORIG_TRAIN_DIR):
    src = os.path.join(ORIG_TRAIN_DIR, fname)
    dst = os.path.join(VAE_TRAIN_DIR, fname)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

print("Original training images copied to working directory.")


# In[6]:


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 16x16
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# In[7]:


# VAE loss
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


# Dataset for single class
class SingleClassDataset(Dataset):
    def __init__(self, csv_file, image_dir, class_idx):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df.iloc[:, class_idx + 1] == 1]
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx, 0])
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


def train_vae(class_idx, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SingleClassDataset(
        csv_file="/kaggle/input/final-project-deep-learning-fall-2025/final_project_resources/train.csv",
        image_dir="/kaggle/input/final-project-deep-learning-fall-2025/final_project_resources/images/train",
        class_idx=class_idx
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ConvVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs in loader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Class {class_idx} | Epoch {epoch+1} | Loss {total_loss/len(dataset):.2f}")

    return model


# In[8]:


def generate_images(model, num_images, save_dir, prefix):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, 128).to(device)
            img = model.decode(z)
            save_image(img, os.path.join(save_dir, f"{prefix}_vae_{i}.png"))


# Train VAEs
vae_G = train_vae(class_idx=1)  # Glaucoma
vae_A = train_vae(class_idx=2)  # AMD

# Generate images to balance classes
generate_images(vae_G, 354, VAE_TRAIN_DIR, "G")
generate_images(vae_A, 375, VAE_TRAIN_DIR, "A")


# In[9]:


df = train_df

new_rows = []

for i in range(354):
    new_rows.append([f"G_vae_{i}.png", 0, 1, 0])

for i in range(375):
    new_rows.append([f"A_vae_{i}.png", 0, 0, 1])

df_vae = pd.concat(
    [df, pd.DataFrame(new_rows, columns=df.columns)],
    ignore_index=True
)

df_vae.to_csv("train_vae.csv", index=False)

print("New training set size:", len(df_vae))


# ### Initializing the Train Data Loader (again)

# In[10]:


# Initialize Loaders
train_df = pd.read_csv("train_vae.csv")
train_loader = DataLoader(ODIRDataset(train_df, "/kaggle/working/images", 'train'), batch_size=32, shuffle=True)


# ## Updated Disease Distribution Visualization

# In[11]:


# =========================
# Dataset Distribution AFTER VAE Augmentation
# =========================

# Columns representing the three major diseases
DISEASE_COLUMNS = ['D', 'G', 'A']

# Calculate the count of each disease in the training set
disease_counts = train_df[DISEASE_COLUMNS].sum()

plt.figure(figsize=(8, 6))
bars = plt.bar(disease_counts.index, disease_counts.values,
               color=['green', 'blue', 'red'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             yval + 10,
             int(yval),
             ha='center',
             va='bottom',
             fontsize=12)

plt.title('Distribution of Diseases in Training Dataset (After VAE Augmentation)')
plt.xlabel('Disease Type')
plt.ylabel('Number of Images')
plt.ylim(0, 550)
plt.grid(axis='y', linestyle='--')

CHART_PATH_VAE = 'disease_distribution_train_vae.png'
plt.savefig(CHART_PATH_VAE)
plt.show()

print(f"VAE-augmented disease distribution chart saved to: {CHART_PATH_VAE}")


# In[12]:


print("Train CSV rows:", len(train_df))
print("Train image files:", len(os.listdir("/kaggle/working/images/train")))

print("Sample image paths:")
for i in range(5):
    print(os.path.join("/kaggle/working/images/train", train_df.iloc[i]['id']))


# In[13]:


def show_split_samples(df, image_root, split_name, mode, n=5):
    """
    df         : dataframe for the split
    image_root: base image directory
    split_name: title for display
    mode       : 'train', 'val', 'offsite_test', 'test_onsite'
    """

    print(f"\n===== {split_name} CSV HEAD =====")
    display(df.head())

    # Resolve image folder exactly as Dataset does
    if mode == 'train':
        img_dir = os.path.join(image_root, 'train')
    elif mode == 'val':
        img_dir = os.path.join(image_root, 'val')
    elif mode in ['test', 'offsite_test']:
        img_dir = os.path.join(image_root, 'offsite_test')
    elif mode == 'test_onsite':
        img_dir = os.path.join(image_root, 'onsite_test')

    print(f"\n===== {split_name} SAMPLE IMAGES =====")

    fig, axes = plt.subplots(1, n, figsize=(15, 4))
    for i in range(n):
        img_path = os.path.join(img_dir, df.iloc[i]['id'])
        img = Image.open(img_path).convert("RGB")
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(df.iloc[i]['id'], fontsize=8)

    plt.tight_layout()
    plt.show()


# In[14]:


# TRAIN (VAE-augmented)
show_split_samples(
    train_df,
    "/kaggle/working/images",
    "TRAIN (VAE-Augmented)",
    mode="train"
)

# VALIDATION
show_split_samples(
    val_df,
    IMAGE_DIR,
    "VALIDATION",
    mode="val"
)

# OFFSITE TEST
show_split_samples(
    offsite_test_df,
    IMAGE_DIR,
    "OFFSITE TEST",
    mode="offsite_test"
)

# ONSITE TEST
onsite_test_df = pd.read_csv(os.path.join(BASE_DIR, 'onsite_test_submission.csv'))

show_split_samples(
    onsite_test_df,
    IMAGE_DIR,
    "ONSITE TEST",
    mode="test_onsite"
)


# # Task 1: Transfer Learning Implementation

# In[15]:


class TransferModel(nn.Module):
    def __init__(self, backbone, num_classes=3):
        super().__init__()
        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'num_features'):  # ViT / Swin
            in_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)

def evaluate_and_print_metrics(model, data_loader, task_name, model_label):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(DEVICE))
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    all_preds, all_labels = np.concatenate(all_preds), np.concatenate(all_labels)
    precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    avg_fscore = fscore.mean()
    
    print(f"\n--- {task_name} Metrics ({model_label}) ---")
    print(f"Average F-score: {avg_fscore:.4f}")
    diseases = ["DR", "Glaucoma", "AMD"]
    for i in range(3):
        print(f"{diseases[i]}: F-score={fscore[i]:.4f}, Precision={precision[i]:.4f}, Recall={recall[i]:.4f}")
    return avg_fscore

def generate_onsite_submission(model, save_path):
    model.eval()
    template = pd.read_csv(os.path.join(BASE_DIR, 'onsite_test_submission.csv'))
    onsite_loader = DataLoader(ODIRDataset(template, IMAGE_DIR, 'test_onsite'), batch_size=32)
    results = []
    with torch.no_grad():
        for inputs, ids in onsite_loader:
            outputs = model(inputs.to(DEVICE))
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            for i in range(len(ids)):
                results.append({'id': ids[i], 'D': preds[i,0], 'G': preds[i,1], 'A': preds[i,2]})
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"Onsite Submission saved to: {save_path}")

# def load_backbone_only(model, checkpoint_filename):
#     path = os.path.join(BASE_DIR, f'pretrained_backbone/{checkpoint_filename}')
#     ckpt = torch.load(path, map_location=DEVICE)
#     for key in ['classifier.1.weight', 'classifier.1.bias', 'fc.weight', 'fc.bias']:
#         if key in ckpt: del ckpt[key]
#     model.load_state_dict(ckpt, strict=False)
#     return model.to(DEVICE)



def load_backbone_only(model, full_checkpoint_path):
    """
    Loads weights from a FULL path and removes classifier keys.
    """
    # Load the checkpoint directly from the path provided
    checkpoint = torch.load(full_checkpoint_path, map_location=DEVICE)
    
    
    mismatched_keys = ['classifier.1.weight', 'classifier.1.bias', 'fc.weight', 'fc.bias']
    for key in mismatched_keys:
        if key in checkpoint:
            del checkpoint[key]

    model.load_state_dict(checkpoint, strict=False)
    return model.to(DEVICE)


# In[16]:


def get_loaded_backbone(model_func, checkpoint_filename):
    """
    Constructs the path once and calls the loader.
    """
    # 1. Instantiate base model
    backbone = model_func(weights=None) 
    
    # 2. Build the FULL path 
    full_path = os.path.join(BASE_DIR, f'pretrained_backbone/{checkpoint_filename}')
    
    # 3. Pass the clean path to the loader
    return load_backbone_only(backbone, full_path)


# ## Task 1.1 - Baseline Evaluation (Offsite Metrics + Onsite File)

# In[17]:


print("Running Task 1.1 Baselines...")

def run_1_1(model_func, ckpt, label):
    model = model_func(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, f'pretrained_backbone/{ckpt}'), map_location=DEVICE))
    evaluate_and_print_metrics(model, offsite_loader, "Task 1.1", label)
    generate_onsite_submission(model, f"{TEAM_ID}_task1-1_{label}.csv")

run_1_1(efficientnet_b0, 'ckpt_efficientnet_ep50.pt', 'eff')
run_1_1(resnet18, 'ckpt_resnet18_ep50.pt', 'res')


# ## Task 1.2: Frozen Backbone, Fine-Tune Classifier Only

# In[18]:


def train_model(model, optimizer, criterion, epochs, model_label, task_label):
    best_f1 = 0
    save_path = f"Arshman-Faizan-Marwa_{task_label}_{model_label}.pt"
    for epoch in range(1, epochs+1):
        model.train()
        for img, lbl in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(img.to(DEVICE)), lbl.to(DEVICE))
            loss.backward(); optimizer.step()
        
        f1 = evaluate_and_print_metrics(model, val_loader, f"Epoch {epoch}", model_label)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
    return save_path


# ## Task 1.2(a): Training and Evaluation (EfficientNet)

# In[19]:


eff_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_efficientnet_ep50.pt')

# --- EfficientNet Task 1.2 ---
model_eff = TransferModel(load_backbone_only(efficientnet_b0(weights=None), eff_ckpt_path)).to(DEVICE)
for param in model_eff.backbone.parameters(): param.requires_grad = False
opt = optim.Adam(model_eff.classifier.parameters(), lr=1e-3)

best_path = train_model(model_eff, opt, nn.BCEWithLogitsLoss(), 25, "EffNet", "task1-2")
model_eff.load_state_dict(torch.load(best_path))
evaluate_and_print_metrics(model_eff, offsite_loader, "Task 1.2 Final", "EffNet")
generate_onsite_submission(model_eff, f"{TEAM_ID}_task1-2_eff.csv")


# ## Task 1.2(b): Training and Evaluation (ResNet18)

# In[20]:


# --- ResNet18 Task 1.2 (Following your EfficientNet Logic) ---
print("Task 1.2 for ResNet18...")
res_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_resnet18_ep50.pt')
# 1. Setup Model: Load ResNet18 backbone, strip head, and move to DEVICE
# This uses the specific ResNet checkpoint: ckpt_resnet18_ep50.pt
model_res = TransferModel(
    load_backbone_only(resnet18(weights=None), res_ckpt_path)
).to(DEVICE)

# 2. FREEZE the ResNet backbone (Crucial for Task 1.2)
for param in model_res.backbone.parameters(): 
    param.requires_grad = False

# 3. Setup Optimizer for the classifier head only
opt_res = optim.Adam(model_res.classifier.parameters(), lr=1e-3)

# 4. Train using existing train_model function
# Running for 25 epochs 
best_path_res = train_model(
    model_res, 
    opt_res, 
    nn.BCEWithLogitsLoss(), 
    25, 
    "ResNet", 
    "task1-2"
)

# 5. Load the best saved weights
model_res.load_state_dict(torch.load(best_path_res))

# 6. Final Evaluation on Offsite Test Set
evaluate_and_print_metrics(model_res, offsite_loader, "Task 1.2 Final", "ResNet")

# 7. Generate Onsite Submission for Kaggle (Target: 61.4)
generate_onsite_submission(model_res, f"{TEAM_ID}_task1-2_res.csv")


# ## Task 1.3 Full fine-tuning

# ## Task 1.3(a): Training and Evaluation (EfficientNet)

# In[ ]:


# --- TASK 1.3: FULL FINE-TUNING ---
print("Task 1.3: Full Fine-Tuning for EfficientNet...")

# 1. Load the model from Task 1.2 Best Weights
eff_backbone = get_loaded_backbone(efficientnet_b0, 'ckpt_efficientnet_ep50.pt')
model_eff_1_3 = TransferModel(eff_backbone).to(DEVICE)
model_eff_1_3.load_state_dict(torch.load('Arshman-Faizan-Marwa_task1-2_EffNet.pt', map_location=DEVICE))

# 2. UNFREEZE EVERYTHING (Crucial for Task 1.3)
for param in model_eff_1_3.parameters():
    param.requires_grad = True

# 3. Setup Optimizer with a MUCH LOWER Learning Rate
optimizer = optim.Adam(model_eff_1_3.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# 4. Train the model

task1_3_best_path = 'Arshman-Faizan-Marwa_task1-3_EffNet.pt'

best_val_f1 = 0.0
for epoch in range(1, 26): # Run for 25 epochs
    start_time = time.time()
    model_eff_1_3.train()
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model_eff_1_3(inputs), labels)
        loss.backward()
        optimizer.step()
        
    # Evaluate on Validation Set
    val_f1 = evaluate_and_print_metrics(model_eff_1_3, val_loader, f'Task 1.3 Val (Ep {epoch})', 'EffNet')
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model_eff_1_3.state_dict(), task1_3_best_path)
        print(f"-> Task 1.3 Checkpoint Saved! Best Val F1: {best_val_f1:.4f}")

# 5. Final Evaluation and Submission
model_eff_1_3.load_state_dict(torch.load(task1_3_best_path))
evaluate_and_print_metrics(model_eff_1_3, offsite_loader, "Task 1.3 Final", "EffNet")
# Generate the CSV for Kaggle
generate_onsite_submission(model_eff_1_3, 'Arshman-Faizan-Marwa_task1-3_eff.csv')


# ## Task 1.3(b): Training and Evaluation (ResNet18)

# In[22]:


# --- TASK 1.3: FULL FINE-TUNING FOR RESNET18 ---
print("Task 1.3: Full Fine-Tuning for ResNet18...")

# 1. Load the architecture and ResNet baseline backbone
# Using the path logic that works for your environment
res_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_resnet18_ep50.pt')
res_backbone = load_backbone_only(resnet18(weights=None), res_ckpt_path)
model_res_1_3 = TransferModel(res_backbone).to(DEVICE)

# 2. LOAD YOUR BEST WEIGHTS FROM RESNET TASK 1.2
# This filename should match the save_path from your ResNet 1.2 run
task1_2_res_weights = 'Arshman-Faizan-Marwa_task1-2_ResNet.pt' 

if os.path.exists(task1_2_res_weights):
    model_res_1_3.load_state_dict(torch.load(task1_2_res_weights, map_location=DEVICE))
    print(f"Successfully loaded ResNet Task 1.2 weights.")
else:
    print(f"ERROR: {task1_2_res_weights} not found. Please check the filename.")

# 3. UNFREEZE EVERYTHING
for param in model_res_1_3.parameters():
    param.requires_grad = True

# 4. Setup Optimizer with a LOW Learning Rate (1e-5 or 2e-5)
# We use a low LR so we don't destroy the pre-trained features
optimizer = optim.Adam(model_res_1_3.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# 5. Training Loop
task1_3_res_best_path = 'Arshman-Faizan-Marwa_task1-3_ResNet.pt'
best_val_f1 = 0.0

for epoch in range(1, 26): # Running for 25 epochs
    model_res_1_3.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model_res_1_3(inputs), labels)
        loss.backward()
        optimizer.step()
        
    # Evaluate on Validation Set
    val_f1 = evaluate_and_print_metrics(model_res_1_3, val_loader, f'Task 1.3 Val (Ep {epoch})', 'ResNet')
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model_res_1_3.state_dict(), task1_3_res_best_path)
        print(f"-> Task 1.3 ResNet Checkpoint Saved! Best Val F1: {best_val_f1:.4f}")

# 6. Final Evaluation and Submission (Target: 78.8)
model_res_1_3.load_state_dict(torch.load(task1_3_res_best_path))
print("\n--- Final Metrics for ResNet Task 1.3 ---")
evaluate_and_print_metrics(model_res_1_3, offsite_loader, "Task 1.3 Final", "ResNet")
generate_onsite_submission(model_res_1_3, f"{TEAM_ID}_task1-3_res.csv")


# # Task 2: Loss Functions

# ## Function: Focal Loss Function (multi-label, logits)

# In[23]:


class FocalLossMultiLabel(nn.Module):
    """
    Multi-label focal loss built on BCE-with-logits.
    Works with logits of shape [B, C] and targets in {0,1} of shape [B, C].

    gamma: focusing parameter (typical: 2.0)
    alpha: balancing factor:
      - float in (0,1): same alpha for all classes (applied to positive term)
      - list/np/torch of shape [C]: per-class alpha for positive term
    reduction: 'mean' or 'sum' or 'none'
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        # BCE per element
        bce_loss = self.bce(logits, targets)  # [B, C]

        # Probabilities
        probs = torch.sigmoid(logits)  # [B, C]
        pt = probs * targets + (1 - probs) * (1 - targets)  # prob of the true label

        # Alpha weighting (optional)
        if self.alpha is None:
            alpha_t = 1.0
        else:
            if isinstance(self.alpha, (float, int)):
                alpha_pos = torch.full_like(targets, float(self.alpha))
            else:
                alpha_tensor = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype).view(1, -1)
                alpha_pos = alpha_tensor.expand_as(targets)
            alpha_t = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)

        focal_factor = (1 - pt).pow(self.gamma)
        loss = alpha_t * focal_factor * bce_loss  # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ## Function: Class-Balanced Loss (re-weight BCE by class frequency)

# In[24]:


class ClassBalancedBCELoss(nn.Module):
    """
    Class-Balanced BCE (multi-label).
    Uses "effective number of samples" weighting (Cui et al.) for *positive labels* per class,
    then applies that weight to the per-class BCE terms.

    beta close to 1.0 (e.g., 0.999) is typical when dataset is not huge.
    """
    def __init__(self, samples_per_class, beta=0.999, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        spc = torch.tensor(samples_per_class, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(torch.tensor(beta), spc)
        weights = (1.0 - beta) / (effective_num + 1e-12)  # [C]
        # Normalize weights to keep loss scale stable
        weights = weights / weights.sum() * len(weights)

        self.register_buffer("pos_weights", weights)  # [C]

    def forward(self, logits, targets):
        # BCE per element [B, C]
        bce_loss = self.bce(logits, targets)

        # Apply class-balanced weight to positive labels only
        # weight matrix: targets==1 gets pos_weights[c], targets==0 gets 1.0
        w = targets * self.pos_weights.view(1, -1) + (1 - targets) * 1.0
        loss = w * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ## Compute class stats from training dataset

# In[25]:


def get_samples_per_class_from_df(df, disease_cols):
    # For multi-label, "samples_per_class" = number of positives for each class
    counts = df[disease_cols].sum().values.astype(np.float32)  # [C]
    # Avoid zeros (shouldn't happen here, but keep safe)
    counts = np.clip(counts, 1.0, None)
    return counts


# ## Grad-CAM Class

# In[26]:


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, class_idx):
        self.model.zero_grad()
        outputs = self.model(x.to(next(self.model.parameters()).device))
        score = outputs[:, class_idx].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


# In[27]:


def get_gradcam_target_layer(model, backbone):
    if backbone == "resnet":
        return model.backbone.layer4[-1]

    elif backbone == "efficientnet":
        # Find last Conv2d layer automatically
        for module in reversed(list(model.backbone.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise RuntimeError("No Conv2d layer found in EfficientNet backbone")

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


# ## GradCAM Attention Regularization

# In[28]:


def attention_regularization(cam):
    cam = cam.view(cam.size(0), -1)
    cam = cam / (cam.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -torch.sum(cam * torch.log(cam + 1e-8), dim=1)
    return entropy.mean()


# In[29]:


def cam_attention_alignment_loss(cam, attn):
    """
    cam:  (1, H, W)
    attn: (1, C, 1, 1)
    """
    cam = cam.unsqueeze(1)                 # (1,1,H,W)
    cam = F.interpolate(cam, size=(1,1), mode="bilinear")
    cam = cam.squeeze()                    # scalar proxy

    attn = attn.mean()                     # scalar proxy

    return (cam - attn).abs()


# ## Training loop: full fine-tuning using a custom loss and GradCAM

# In[30]:


def train_full_finetune_with_loss(
    model,
    loss_fn,
    epochs,
    lr,
    model_label,
    task_label,
    save_path,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            # ---------- AMP FORWARD ----------
            with torch.amp.autocast(device_type=DEVICE.type):
                logits = model(inputs)
                loss = loss_fn(logits, labels)

                # ===== Grad-CAM Guided Learning (Task 4) =====
                with torch.enable_grad():
                    # Get predicted class per sample
                    pred_classes = torch.argmax(torch.sigmoid(logits), dim=1)
                
                    # Identify target layer ONCE
                    backbone = model.backbone
                    if hasattr(backbone, "layer4"):            # ResNet
                        target_layer = backbone.layer4[-1]
                    elif hasattr(backbone, "features"):        # EfficientNet
                        target_layer = backbone.features[-1]
                    else:
                        target_layer = None
                
                    if target_layer is not None:
                        cam_generator = GradCAM(model, target_layer)
                
                        cam_loss = 0.0
                        valid_samples = 0
                
                        for i in range(inputs.size(0)):
                            cls_idx = pred_classes[i].item()
                
                            # Skip AMD if you want (optional but defensible)
                            if cls_idx == 2:
                                continue
                
                            cam = cam_generator.generate(inputs[i:i+1], cls_idx)
                            # --- CAM–Attention alignment ---
                            if hasattr(model, "attn") and hasattr(model.attn, "last_attention"):
                                attn_map = model.attn.last_attention
                                cam_loss += cam_attention_alignment_loss(cam, attn_map)
                            else:
                                cam_loss += attention_regularization(cam)
                            valid_samples += 1
                
                        if valid_samples > 0:
                            cam_loss = cam_loss / valid_samples
                            loss = loss + 0.01 * cam_loss


            # ---------- AMP BACKWARD ----------
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_f1 = evaluate_and_print_metrics(model, val_loader, f"{task_label} Val (Ep {epoch})", model_label)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"-> Saved best checkpoint to {save_path} | Best Val F1: {best_val_f1:.4f}")

    return save_path


# ## Shared statistics for losses

# In[31]:


# --- shared statistics for losses ---
pos_counts = train_df[DISEASE_COLUMNS].sum().values.astype(np.float32)
total = float(len(train_df))
pos_rate = pos_counts / total

# Per-class alpha heuristic (more weight for minority positives), normalized
alpha_per_class = (1.0 - pos_rate)
alpha_per_class = (alpha_per_class / alpha_per_class.sum()) * len(alpha_per_class)

samples_per_class = get_samples_per_class_from_df(train_df, DISEASE_COLUMNS)


# ## Task 2.1(a): Training and Evaluation (EfficientNet)

# In[32]:


# -----------------------------
# TASK 2.1: Focal Loss (EffNet)
# -----------------------------
print("\n--- TASK 2.1: Focal Loss (EffNet) ---")

eff_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_efficientnet_ep50.pt')
eff_backbone = load_backbone_only(efficientnet_b0(weights=None), eff_ckpt_path)
model_eff_2_1 = TransferModel(eff_backbone).to(DEVICE)

for p in model_eff_2_1.parameters():
    p.requires_grad = True

criterion_focal = FocalLossMultiLabel(gamma=1.0, alpha=0.5, reduction="mean")

task2_1_eff_best_path = 'Arshman-Faizan-Marwa_task2-1_EffNet.pt'
train_full_finetune_with_loss(
    model=model_eff_2_1,
    loss_fn=criterion_focal,
    epochs=25,
    lr=2e-5,
    model_label="EffNet",
    task_label="Task 2.1 (Focal)",
    save_path=task2_1_eff_best_path
)

model_eff_2_1.load_state_dict(torch.load(task2_1_eff_best_path, map_location=DEVICE))
print("\n--- Task 2.1 Final Metrics (Offsite) EffNet ---")
evaluate_and_print_metrics(model_eff_2_1, offsite_loader, "Task 2.1 Final (Offsite)", "EffNet")
generate_onsite_submission(model_eff_2_1, f"{TEAM_ID}_task2-1_eff.csv")


# ## Task 2.1(b): Training and Evaluation (ResNet18)

# In[33]:


# -----------------------------
# TASK 2.1: Focal Loss (ResNet)
# -----------------------------
print("\n--- TASK 2.1: Focal Loss (ResNet) ---")

res_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_resnet18_ep50.pt')
res_backbone = load_backbone_only(resnet18(weights=None), res_ckpt_path)
model_res_2_1 = TransferModel(res_backbone).to(DEVICE)

for p in model_res_2_1.parameters():
    p.requires_grad = True

criterion_focal = FocalLossMultiLabel(gamma=1.0, alpha=0.5, reduction="mean")

task2_1_res_best_path = 'Arshman-Faizan-Marwa_task2-1_ResNet.pt'
train_full_finetune_with_loss(
    model=model_res_2_1,
    loss_fn=criterion_focal,
    epochs=25,
    lr=2e-5,
    model_label="ResNet",
    task_label="Task 2.1 (Focal)",
    save_path=task2_1_res_best_path
)

model_res_2_1.load_state_dict(torch.load(task2_1_res_best_path, map_location=DEVICE))
print("\n--- Task 2.1 Final Metrics (Offsite) ResNet ---")
evaluate_and_print_metrics(model_res_2_1, offsite_loader, "Task 2.1 Final (Offsite)", "ResNet")
generate_onsite_submission(model_res_2_1, f"{TEAM_ID}_task2-1_res.csv")


# ## Task 2.2(a): Training and Evaluation (EfficientNet)

# In[34]:


# --------------------------------------
# TASK 2.2: Class-Balanced BCE (EffNet)
# --------------------------------------
print("\n--- TASK 2.2: Class-Balanced BCE (EffNet) ---")

eff_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_efficientnet_ep50.pt')
eff_backbone = load_backbone_only(efficientnet_b0(weights=None), eff_ckpt_path)
model_eff_2_2 = TransferModel(eff_backbone).to(DEVICE)

for p in model_eff_2_2.parameters():
    p.requires_grad = True

criterion_cb = ClassBalancedBCELoss(samples_per_class=samples_per_class, beta=0.999, reduction="mean").to(DEVICE)

task2_2_eff_best_path = 'Arshman-Faizan-Marwa_task2-2_EffNet.pt'
train_full_finetune_with_loss(
    model=model_eff_2_2,
    loss_fn=criterion_cb,
    epochs=25,
    lr=2e-5,
    model_label="EffNet",
    task_label="Task 2.2 (ClassBalanced)",
    save_path=task2_2_eff_best_path
)

model_eff_2_2.load_state_dict(torch.load(task2_2_eff_best_path, map_location=DEVICE))
print("\n--- Task 2.2 Final Metrics (Offsite) EffNet ---")
evaluate_and_print_metrics(model_eff_2_2, offsite_loader, "Task 2.2 Final (Offsite)", "EffNet")
generate_onsite_submission(model_eff_2_2, f"{TEAM_ID}_task2-2_eff.csv")


# ## Task 2.2(b): Training and Evaluation (ResNet18)

# In[35]:


# --------------------------------------
# TASK 2.2: Class-Balanced BCE (ResNet)
# --------------------------------------
print("\n--- TASK 2.2: Class-Balanced BCE (ResNet) ---")

res_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_resnet18_ep50.pt')
res_backbone = load_backbone_only(resnet18(weights=None), res_ckpt_path)
model_res_2_2 = TransferModel(res_backbone).to(DEVICE)

for p in model_res_2_2.parameters():
    p.requires_grad = True

criterion_cb = ClassBalancedBCELoss(samples_per_class=samples_per_class, beta=0.999, reduction="mean").to(DEVICE)

task2_2_res_best_path = 'Arshman-Faizan-Marwa_task2-2_ResNet.pt'
train_full_finetune_with_loss(
    model=model_res_2_2,
    loss_fn=criterion_cb,
    epochs=25,
    lr=2e-5,
    model_label="ResNet",
    task_label="Task 2.2 (ClassBalanced)",
    save_path=task2_2_res_best_path
)

model_res_2_2.load_state_dict(torch.load(task2_2_res_best_path, map_location=DEVICE))
print("\n--- Task 2.2 Final Metrics (Offsite) ResNet ---")
evaluate_and_print_metrics(model_res_2_2, offsite_loader, "Task 2.2 Final (Offsite)", "ResNet")
generate_onsite_submission(model_res_2_2, f"{TEAM_ID}_task2-2_res.csv")


# # Task 3: Attention Mechanisms (SE + Multi-head Attention)

# ## Function: Squeeze-and-Excitation

# In[36]:


class SEBlock(nn.Module):
    """
    Channel attention (Squeeze-and-Excitation).
    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )
        self.last_attention = None

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)         # (B, C)
        w = self.fc(s).view(b, c, 1, 1)     # (B, C, 1, 1)
        self.last_attention = w.detach()
        return x * w


# ## Function: Multi-head Attention

# In[37]:


class SpatialMHABlock(nn.Module):
    """
    Self-attention over spatial tokens.
    Treats each spatial position as a token with embedding dim = C.
    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, HW, C)
        tokens = x.view(b, c, h * w).transpose(1, 2)
        tokens_n = self.norm(tokens)
        attn_out, _ = self.mha(tokens_n, tokens_n, tokens_n, need_weights=False)
        tokens = tokens + self.dropout(attn_out)  # residual
        # (B, HW, C) -> (B, C, H, W)
        return tokens.transpose(1, 2).view(b, c, h, w)


# ## Attention-enabled models for ResNet18 
# ### (uses your pretrained backbone checkpoints via load_backbone_only())

# In[38]:


class ResNet18WithAttention(nn.Module):
    def __init__(self, resnet_backbone: nn.Module, attention: str = "none",
                 num_classes: int = 3, se_reduction: int = 16, mha_heads: int = 8):
        super().__init__()
        self.backbone = resnet_backbone  # already loaded with your checkpoint
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )

        c = 512  # ResNet18 layer4 output channels
        if attention.lower() == "se":
            self.attn = SEBlock(c, reduction=se_reduction)
        elif attention.lower() == "mha":
            self.attn = SpatialMHABlock(c, num_heads=mha_heads, dropout=0.0)
        else:
            self.attn = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)

        # same style head as your TransferModel
        self.classifier = nn.Sequential(
            nn.Linear(c, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        f = self.feature_extractor(x)       # (B, 512, H, W)
        f = self.attn(f)
        f = self.pool(f).flatten(1)         # (B, 512)
        return self.classifier(f)


# ## Attention-enabled models for EfficientNetB0 
# ### (uses your pretrained backbone checkpoints via load_backbone_only())

# In[39]:


class EfficientNetB0WithAttention(nn.Module):
    def __init__(self, eff_backbone: nn.Module, attention: str = "none",
                 num_classes: int = 3, se_reduction: int = 16, mha_heads: int = 8):
        super().__init__()
        self.backbone = eff_backbone  # already loaded with your checkpoint
        self.features = self.backbone.features

        c = 1280  # EfficientNetB0 final feature channels
        if attention.lower() == "se":
            self.attn = SEBlock(c, reduction=se_reduction)
        elif attention.lower() == "mha":
            self.attn = SpatialMHABlock(c, num_heads=mha_heads, dropout=0.0)
        else:
            self.attn = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(c, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        f = self.features(x)                # (B, 1280, H, W)
        f = self.attn(f)
        f = self.pool(f).flatten(1)         # (B, 1280)
        f = self.dropout(f)
        return self.classifier(f)


# ## Loss Choice

# In[40]:


# ------------------------------------------------------------
# Loss choice: prefer your ClassBalancedBCELoss if available,
# else fallback to BCEWithLogitsLoss.
# ------------------------------------------------------------
def get_task4_loss():
    # If you already defined ClassBalancedBCELoss in Task 2, use it.
    if "ClassBalancedBCELoss" in globals():
        samples_per_class = train_df[DISEASE_COLUMNS].sum().values.astype(np.float32)
        return ClassBalancedBCELoss(samples_per_class=samples_per_class, beta=0.999, reduction="mean").to(DEVICE)
    return nn.BCEWithLogitsLoss()


# # Task 4: Fine-tuning with GradCAM and backbone networks like Swin and Vision Transformer

# ## Task 4.1(a): Training and Evaluation (GradCAM improved SE Attention (EffNet))

# In[41]:


print("\n--- TASK 4.1(a): GradCAM improved SE Attention (EffNet) ---")

task4_criterion = get_task4_loss()

eff_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_efficientnet_ep50.pt')
eff_backbone = load_backbone_only(efficientnet_b0(weights=None), eff_ckpt_path)

model_se_eff_4_1 = EfficientNetB0WithAttention(
    eff_backbone=eff_backbone,
    attention="se",
    num_classes=3,
    se_reduction=16,
    mha_heads=8
).to(DEVICE)

for p in model_se_eff_4_1.parameters():
    p.requires_grad = True

task4_1_se_eff_best_path = "Arshman-Faizan-Marwa_task4-1_SE_EffNet.pt"

train_full_finetune_with_loss(
    model=model_se_eff_4_1,
    loss_fn=task4_criterion,
    epochs=25,
    lr=2e-5,
    model_label="EffNet",
    task_label="Task 4.1(a) SE",
    save_path=task4_1_se_eff_best_path
)

model_se_eff_4_1.load_state_dict(torch.load(task4_1_se_eff_best_path, map_location=DEVICE))
evaluate_and_print_metrics(model_se_eff_4_1, offsite_loader, "Task 4.1(a) Final Offsite", "SE EffNet")
generate_onsite_submission(model_se_eff_4_1, f"{TEAM_ID}_task4-1_se_eff.csv")


# ## Task 4.1(b): Training and Evaluation (GradCAM improved SE Attention (ResNet))

# In[42]:


print("\n--- TASK 4.1(b): GradCAM improved SE Attention (ResNet) ---")

task4_criterion = get_task4_loss()

res_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_resnet18_ep50.pt')
res_backbone = load_backbone_only(resnet18(weights=None), res_ckpt_path)

model_se_res_4_1 = ResNet18WithAttention(
    resnet_backbone=res_backbone,
    attention="se",
    num_classes=3,
    se_reduction=16,
    mha_heads=8
).to(DEVICE)

for p in model_se_res_4_1.parameters():
    p.requires_grad = True

task4_1_se_res_best_path = "Arshman-Faizan-Marwa_task4-1_SE_ResNet.pt"

train_full_finetune_with_loss(
    model=model_se_res_4_1,
    loss_fn=task4_criterion,
    epochs=25,
    lr=2e-5,
    model_label="ResNet",
    task_label="Task 4.1(b) SE",
    save_path=task4_1_se_res_best_path
)

model_se_res_4_1.load_state_dict(torch.load(task4_1_se_res_best_path, map_location=DEVICE))
evaluate_and_print_metrics(model_se_res_4_1, offsite_loader, "Task 4.1(b) Final Offsite", "SE ResNet")
generate_onsite_submission(model_se_res_4_1, f"{TEAM_ID}_task4-1_se_res.csv")


# ## Task 4.1(c): Training and Evaluation (GradCAM improved MHA (EffNet))

# In[43]:


print("\n--- TASK 4.1(c): GradCAM improved Multi-Head Attention (EffNet) ---")

task4_criterion = get_task4_loss()

eff_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_efficientnet_ep50.pt')
eff_backbone = load_backbone_only(efficientnet_b0(weights=None), eff_ckpt_path)

model_mha_eff_4_1 = EfficientNetB0WithAttention(
    eff_backbone=eff_backbone,
    attention="mha",
    num_classes=3,
    se_reduction=16,
    mha_heads=8
).to(DEVICE)

for p in model_mha_eff_4_1.parameters():
    p.requires_grad = True

task4_1_mha_eff_best_path = "Arshman-Faizan-Marwa_task4-1_MHA_EffNet.pt"

train_full_finetune_with_loss(
    model=model_mha_eff_4_1,
    loss_fn=task4_criterion,
    epochs=25,
    lr=2e-5,
    model_label="EffNet",
    task_label="Task 4.1(c) MHA",
    save_path=task4_1_mha_eff_best_path
)

model_mha_eff_4_1.load_state_dict(torch.load(task4_1_mha_eff_best_path, map_location=DEVICE))
evaluate_and_print_metrics(model_mha_eff_4_1, offsite_loader, "Task 4.1(c) Final Offsite", "MHA EffNet")
generate_onsite_submission(model_mha_eff_4_1, f"{TEAM_ID}_task4-1_mha_eff.csv")


# ## Task 4.1(d): Training and Evaluation (GradCAM improved MHA (ResNet))

# In[44]:


print("\n--- TASK 4.1(d): GradCAM improved Multi-Head Attention (ResNet) ---")

task4_criterion = get_task4_loss()

res_ckpt_path = os.path.join(BASE_DIR, 'pretrained_backbone/ckpt_resnet18_ep50.pt')
res_backbone = load_backbone_only(resnet18(weights=None), res_ckpt_path)

model_mha_res_4_1 = ResNet18WithAttention(
    resnet_backbone=res_backbone,
    attention="mha",
    num_classes=3,
    se_reduction=16,
    mha_heads=8
).to(DEVICE)

for p in model_mha_res_4_1.parameters():
    p.requires_grad = True

task4_1_mha_res_best_path = "Arshman-Faizan-Marwa_task4-1_MHA_ResNet.pt"

train_full_finetune_with_loss(
    model=model_mha_res_4_1,
    loss_fn=task4_criterion,
    epochs=25,
    lr=2e-5,
    model_label="ResNet",
    task_label="Task 4.1(d) MHA",
    save_path=task4_1_mha_res_best_path
)

model_mha_res_4_1.load_state_dict(torch.load(task4_1_mha_res_best_path, map_location=DEVICE))
evaluate_and_print_metrics(model_mha_res_4_1, offsite_loader, "Task 4.1(d) Final Offsite", "ResNet")
generate_onsite_submission(model_mha_res_4_1, f"{TEAM_ID}_task4-1_mha_res.csv")


# ## Task 4.1: Grad-CAM Visualization

# In[45]:


import cv2
def visualize_gradcam(model, backbone, dataset, device, num_samples=3):
    model.eval()

    target_layer = get_gradcam_target_layer(model, backbone)
    if target_layer is None:
        print("Grad-CAM visualization skipped (Transformer backbone).")
        return

    cam_generator = GradCAM(model, target_layer)

    class_names = ["DR", "Glaucoma", "AMD"]

    for i in range(num_samples):
        img, label = dataset[i]
        x = img.unsqueeze(0).to(device)

        # 1️⃣ Forward pass to get predictions
        with torch.no_grad():
            outputs = model(x)

        # 2️⃣ Select most confident predicted class
        class_idx = torch.argmax(torch.sigmoid(outputs), dim=1).item()
        cls_name = class_names[class_idx]

        # 3️⃣ Generate Grad-CAM for that class
        cam = cam_generator.generate(x, class_idx)[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (img.shape[2], img.shape[1]))

        img_np = img.permute(1,2,0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = heatmap / 255.0

        overlay = 0.5 * img_np + 0.5 * heatmap

        plt.figure(figsize=(4,4))
        plt.title(f"Pred: {cls_name} | GT={int(label[class_idx])}")
        plt.imshow(overlay)
        plt.axis("off")
        plt.show()


# In[46]:


val_dataset = ODIRDataset(val_df, IMAGE_DIR, 'val')


# In[47]:


visualize_gradcam(
    model_se_eff_4_1,
    backbone="efficientnet",
    dataset=val_dataset,
    device=DEVICE,
    num_samples=3
)


# In[50]:


visualize_gradcam(
    model_se_res_4_1,
    backbone="resnet",
    dataset=val_dataset,
    device=DEVICE,
    num_samples=3
)


# In[51]:


visualize_gradcam(
    model_mha_eff_4_1,
    backbone="efficientnet",
    dataset=val_dataset,
    device=DEVICE,
    num_samples=3
)


# In[52]:


visualize_gradcam(
    model_mha_res_4_1,
    backbone="resnet",
    dataset=val_dataset,
    device=DEVICE,
    num_samples=3
)


# ## Task 4.2 (a): Training and Evaluation (Vision Transformer)

# In[ ]:


print("Task 4: Vision Transformer")

vit_backbone = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    img_size=256,
    num_classes=0
)

model_vit = TransferModel(vit_backbone).to(DEVICE)

opt = optim.Adam(model_vit.parameters(), lr=2e-5)

best_path = train_full_finetune_with_loss(
    model=model_vit,
    loss_fn=nn.BCEWithLogitsLoss(),
    epochs=25,
    lr=2e-5,
    model_label="ViT",
    task_label="Task 4",
    save_path="Arshman-Faizan-Marwa_task4-2_ViT.pt"
)

model_vit.load_state_dict(torch.load(best_path, map_location=DEVICE))
evaluate_and_print_metrics(model_vit, offsite_loader, "Task 4.2a ViT Offsite", "ViT")
generate_onsite_submission(model_vit, f"{TEAM_ID}_task4-2_vit.csv")


# ## Task 4.2 (b): Training and Evaluation (Swin Transformer)

# In[ ]:


print("Task 4: Swin Transformer")

swin_backbone = timm.create_model(
    "swin_small_patch4_window7_224",
    pretrained=True,
    img_size=256,
    num_classes=0,
    global_pool="avg"
)

swin_backbone.set_grad_checkpointing(True)

model_swin = TransferModel(swin_backbone).to(DEVICE)

for name, param in model_swin.backbone.named_parameters():
    if "layers.0" in name or "layers.1" in name:
        param.requires_grad = False

opt = optim.Adam(model_swin.parameters(), lr=2e-5)

best_path = train_full_finetune_with_loss(
    model=model_swin,
    loss_fn=nn.BCEWithLogitsLoss(),
    epochs=25,
    lr=2e-5,
    model_label="Swin",
    task_label="Task 4",
    save_path="Arshman-Faizan-Marwa_task4-2_Swin.pt"
)

model_swin.load_state_dict(torch.load(best_path, map_location=DEVICE))
evaluate_and_print_metrics(model_swin, offsite_loader, "Task 4.2b Swin Offsite", "Swin")
generate_onsite_submission(model_swin, f"{TEAM_ID}_task4-2_swin.csv")

