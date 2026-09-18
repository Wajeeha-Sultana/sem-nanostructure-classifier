# SEM Nanostructure-Type Classifier

A Convolutional Neural Network (CNN) that classifies Scanning Electron Microscopy (SEM) images
into four nanostructure morphology categories — built as a NAVTTC "Artificial Intelligence
(Machine Learning & Deep Learning)" CNN capstone project.

## Problem

SEM imaging is the standard technique for characterizing the morphology of synthesized
nanomaterials (particles, nanowires, porous networks, patterned surfaces, etc.). Classifying
SEM micrographs by morphology is currently a manual, visual task for researchers. This project
trains a CNN to automate that first-pass classification.

## Dataset

- Source: [`motiurinfo/SEM-Dataset-500`](https://github.com/motiurinfo/SEM-Dataset-500), a
  500-image curated high-quality subset of the NFFA-EUROPE annotated SEM dataset.
- Original citation: Aversa, R., Modarres, M., Cozzini, S. *et al.* "The first annotated set of
  scanning electron microscopy images for nanoscience." *Scientific Data* 5, 180172 (2018).
  https://doi.org/10.1038/sdata.2018.172
- 435 usable images across 4 classes (imbalanced: 46–153 images/class), organized under
  `data_resized/<class_name>/`. Class names were assigned by the project author after visually
  inspecting each numeric source group, since the 500-image subset does not publish per-class
  names of its own.

| Class | # Images | Description |
|---|---|---|
| `nanowire_fibre_mesh` | 115 | Dense mat of tangled nanowire/fibre strands |
| `particle_coated_surface` | 46 | Fine granular/particle-like surface texture |
| `patterned_structure` | 153 | Sharp-edged, geometric lithographic/electrode pattern |
| `porous_fibrous_network` | 121 | Fine fibrous mesh with visible pores |

## Model

A compact 4-block CNN (Conv2D + MaxPooling ×4, 16→32→64→64 filters) trained from scratch at
128×128 grayscale resolution, with data augmentation and class weighting to handle the small,
imbalanced dataset. See `Assignment8_YourName.ipynb` for the full, executed pipeline
(preprocessing, EDA, training, evaluation).

**Test set result:** ~71% accuracy across 4 classes (see the notebook for the full
classification report and confusion matrix).

## Repository contents

- `Assignment8_YourName.ipynb` — full project notebook (preprocessing → CNN → training → evaluation)
- `data_resized/` — the 435-image dataset, organized by class folder
- `sem_nanostructure_cnn.keras` — the trained model
- `app.py` — Streamlit web app for interactive predictions
- `requirements.txt` — Python dependencies

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), upload an SEM
image, and view the predicted class and per-class confidence.

## Deployment

Deployed on Streamlit Community Cloud — live link in `deliverable_links.txt`.

## Limitations

Trained on only 435 images (46 in the smallest class) with no pretrained-weight transfer
learning (no internet access to ImageNet weights during development), so this is a coursework
/ portfolio demonstration of the CNN workflow rather than a production-validated
materials-characterization tool. See the notebook's evaluation section for a discussion of
per-class performance and next steps.
