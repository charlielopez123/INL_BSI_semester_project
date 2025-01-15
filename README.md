# Brain–Spine Interface (BSI) Neural Decoding

This repository contains the code and resources developed for the **BSI Neural Decoding** project [View paper (PDF)](Report_INL.pdf). The project focuses on decoding upper-body movements using electrocorticography (ECoG) signals with a Vision Transformer (ViT) architecture. The work emphasizes feature engineering and aims to advance brain–machine interfaces (BMIs) for restoring movement in individuals with chronic paralysis.

---

## Project Overview

This project builds upon the foundational work of [Wagner et al.](https://www.nature.com/articles/s41586-023-06094-5), extending it to upperbody movement classification. Key objectives include:

- Developing a transformer-based neural decoding model.
- Exploring feature engineering using FFT, DWT, and CWT.
- Investigating dimensionality reduction techniques like PCA and forward selection.
---

## Features

- **Preprocessing Pipeline**:
  - Bandpass filtering, notch filtering, and common average referencing (CAR) using MNE filters.
- **Feature Extraction**:
  - Fast Fourier Transform (FFT)
  - Discrete Wavelet Transform (DWT)
  - Continuous Wavelet Transform (CWT)
- **Model Architecture**:
  - Vision Transformer (ViT) tailored for ECoG decoding.
- **Feature Selection**:
  - PCA and forward selection for dimensionality reduction.
- **Performance Metrics**:
  - Accuracy, weighted F1 score, and averaged trace of the confusion matrix.

---

## Vision Transformer Models and Testing Workflow
### Repository Overview

This repository contains implementations of Vision Transformer (ViT) models for various tasks, alongside extensions and modifications to the original architecture. It includes:

- The workspace for the base Vision Transformer model (i.e the **original model**) as described in the [paper](Report_INL.pdf), including the shell scripts to run the different models
- The workspace for the BSIT model forked from [XYHZJU/Digital_Bridge_Torch](https://github.com/XYHZJU/Digital_Bridge_Torch) repository, including the shell scripts to run the different models

---
## Running the code

To run the Vision Transformer models and experiments, you must execute the provided shell scripts.

Each shell script supports customizable arguments to adjust model parameters, datasets, or training configurations. To modify these arguments:
