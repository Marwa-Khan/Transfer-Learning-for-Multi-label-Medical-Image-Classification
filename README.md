# 521153S-3006-DL-Transfer-Learning-for-Multi-label-Medical-Image-Classification

# ODIR Multi‑Label Retinal Disease Classification (DR / Glaucoma / AMD)

## Overview
This project trains multi‑label classifiers on the ODIR fundus dataset for **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Age‑related Macular Degeneration (AMD)**. The work progresses through Tasks 1–3 (transfer learning, loss functions, attention) and Task 4 extensions (VAE augmentation, transformer backbones, and GradCAM analysis / GradCAM‑guided training).

## Notebooks (what each one does)

### 1) `Arshman-Faizan-Marwa_task1-task2-task3.ipynb`
Runs **Tasks 1–3 only**:
- Task 1: transfer learning (frozen backbone vs full fine‑tuning)
- Task 2: imbalance-aware losses
- Task 3: attention mechanisms  
No VAE augmentation and no Task‑4 modules are included here.

### 2) `Arshman-Faizan-Marwa_vae-imagegeneration.ipynb`
Implements **VAE-based data augmentation (Task 4a)** and then runs Tasks 1–3 again in the same order. The key difference is that training is performed using the **augmented image set** rather than the original training set.

### 3) `Arshman-Faizan-Marwa_swine-vision-backbone.ipynb`
Extension of the VAE notebook. Adds **two additional backbones** (ViT and Swin) as Task 4b variants. These experiments reuse the Task‑3 pipeline so the chosen loss setup and attention mechanisms are already “built into” the model flow used for training/evaluation.

### 4) `Arshman-Faizan-Marwa_gradcam-visualizations.ipynb`
Further extension of the Swin/ViT notebook, but focused on **GradCAM visualization only** (the Swin/ViT training snippets are commented out to reduce compute/memory usage). This notebook generates heatmaps for models trained in Tasks 1–3 to support qualitative comparison and interpretability checks. It **does not** implement GradCAM‑guided training.

### 5) `Arshman-Faizan-Marwa_gradcam-attentionmap.ipynb`
Extension of the visualization notebook that implements **GradCAM‑guided training (Task 4c)**. It takes Task‑3 models (loss + attention already applied) and further trains them with a CAM-based regularizer so that both predictions and heatmaps reflect GradCAM guidance.

## Outputs
- `*.pt`: saved model checkpoints (trained weights).
- `*.csv`: prediction files used for **on‑site test submission**.

Both checkpoints and CSVs follow a consistent naming convention based on task/variant so runs can be identified from filenames.

## Recommended workflow (for a new user)
1. Start with `Arshman-Faizan-Marwa_task1-task2-task3.ipynb` to reproduce Tasks 1–3 baseline results.
2. Run `Arshman-Faizan-Marwa_vae-imagegeneration.ipynb` to reproduce the same pipeline using VAE‑augmented training images.
3. Run `Arshman-Faizan-Marwa_swine-vision-backbone.ipynb` to evaluate ViT and Swin backbones.
4. Run `Arshman-Faizan-Marwa_gradcam-visualizations.ipynb` to generate GradCAM heatmaps for Task 1–3 models.
5. Run `Arshman-Faizan-Marwa_gradcam-attentionmap.ipynb` to train and evaluate the GradCAM‑guided variant.

## Python source (.py requirement)
To satisfy the “single .py source file” requirement, the final notebook `Arshman-Faizan-Marwa_gradcam-attentionmap.ipynb` was converted to Python (conversion only; not manually rewritten). The conversion was done with:

`jupyter nbconvert --to script Arshman-Faizan-Marwa_gradcam-attentionmap.ipynb`

Note: the exported script reflects the code from the notebook, and the notebook’s training runs were performed on the augmented dataset (not the original training folder).
