# Brain–Spine Interface (BSI) Neural Decoding

This repository contains the code and resources developed for the **BSI Neural Decoding** project. The project focuses on decoding upper-body movements using electrocorticography (ECoG) signals with a Vision Transformer (ViT) architecture. The work emphasizes feature engineering and aims to advance brain–machine interfaces (BMIs) for restoring movement in individuals with chronic paralysis.
[Download the paper (PDF)](Report_INL.pdf)

---

## Project Overview

This project builds upon the foundational work of Wagner et al., extending it to upper-body movement classification. Key objectives include:

- Developing a transformer-based neural decoding model.
- Exploring feature engineering using FFT, DWT, and CWT.
- Investigating dimensionality reduction techniques like PCA and forward selection.
- Optimizing the model for hardware efficiency and real-time implementation.

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
