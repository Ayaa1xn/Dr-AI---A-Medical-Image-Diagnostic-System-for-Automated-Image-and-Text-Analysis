# Dr. AI: Multimodal Medical Diagnostic Assistant

Dr. AI is a web-based medical AI assistant designed to make diagnostic information easier to understand for everyday users. It combines deep learning based medical image classification, Grad-CAM visual explanations, OCR powered report extraction, and large language model based report summarization inside a Flask application.

The goal of Dr. AI is not to replace doctors, clinics, hospitals, or formal medical diagnosis. Instead, the project aims to reduce friction in the diagnostic journey by helping users get an initial, understandable explanation of medical scans and reports before or between clinical consultations. Many patients receive CT scans, MRI scans, prescriptions, lab reports, and other clinical documents filled with medical terminology that is difficult to interpret. Dr. AI helps simplify that language, highlight potentially important findings, and provide a basic understanding of whether a report may require medical follow-up.

> **Medical disclaimer:** Dr. AI is an academic/research prototype and decision-support tool. It is not a certified medical device and must not be used as a substitute for professional clinical judgment, diagnosis, or treatment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Research Motivation](#research-motivation)
- [Research Gap and Technical Contribution](#research-gap-and-technical-contribution)
- [System Architecture](#system-architecture)
- [Machine Learning Models](#machine-learning-models)
- [Datasets](#datasets)
- [Algorithmic Approach](#algorithmic-approach)
- [Training Methodology](#training-methodology)
- [Image Processing Pipeline](#image-processing-pipeline)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Medical Report Analyzer](#medical-report-analyzer)
- [Chatbot Assistant](#chatbot-assistant)
- [Evaluation and Results](#evaluation-and-results)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Implementation Details](#implementation-details)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Application Routes](#application-routes)
- [Security and Production Notes](#security-and-production-notes)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Author](#author)

---

## Project Overview

Dr. AI is an integrated diagnostic assistant with three main modules:

1. **Brain Tumor Detection**
   - Classifies brain MRI scans into four categories:
     - No Tumor
     - Pituitary Tumor
     - Meningioma Tumor
     - Glioma Tumor

2. **Lung Cancer Detection**
   - Classifies lung scan images into four categories:
     - Squamous Cell Carcinoma
     - Adenocarcinoma Cell
     - Large Cell Carcinoma
     - Normal Cell

3. **Medical Report Analyzer**
   - Accepts medical documents such as PDFs, scanned reports, image reports, prescriptions, and text files.
   - Extracts text using PDF parsing and OCR.
   - Sends the extracted content to an LLM pipeline.
   - Returns a structured, simplified, user-friendly medical summary.

The application also includes a floating medical chatbot and downloadable PDF report generation.

---

## Key Features

- Web-based diagnostic assistant built with Flask.
- Custom ResNet50-style CNN models trained for grayscale medical scans.
- Brain MRI and lung cancer image classification.
- Grad-CAM++ visualizations to show model attention regions.
- Medical report upload and simplification.
- PDF text extraction using `pdfminer.six`.
- OCR fallback using `EasyOCR`.
- LLM-assisted report summarization using OpenAI chat completions.
- Downloadable medical summary PDF using ReportLab.
- Chatbot interface for health-related questions.
- User-facing web interface with upload, prediction, result, and report pages.

---

## Research Motivation

Medical reports and diagnostic scans are often difficult for non-specialists to understand. A patient may receive a scan or report containing terms such as "meningioma", "adenocarcinoma", "abnormal soft tissue density", "elevated inflammatory markers", or "clinical correlation advised" without knowing what those terms mean or how urgent the situation is.

This creates a repeated cycle:

1. The patient visits a hospital or diagnostic center.
2. The patient receives a scan, prescription, or lab report.
3. The patient must then visit another clinician simply to understand the terminology.
4. Further visits may be needed for follow-up, clarification, or reassurance.

Dr. AI was designed to reduce that informational burden. It provides a first-level explanation of scans and reports in simpler language, helping users understand what may be serious, what may be routine, and what questions they should ask a doctor.

The project is intentionally positioned as a supportive tool, not a replacement for professional healthcare. Its purpose is to ease the burden on patients and clinicians by making medical information more understandable and accessible.

---

## Research Gap and Technical Contribution

Many medical image classification systems adapt CNN architectures originally designed for natural RGB images. Architectures such as ResNet50, VGG16, EfficientNet, and Inception were commonly trained or initialized on ImageNet-style three-channel RGB images. However, medical scans such as MRI and CT images are usually grayscale and rely on intensity, structure, texture, and contrast rather than natural color features.

Common approaches include:

- Fine-tuning pretrained RGB models on medical images.
- Duplicating grayscale scans into three identical RGB channels.
- Modifying only the first convolutional layer of a pretrained architecture.

These approaches can introduce problems:

- **Redundant computation:** duplicating grayscale data into three channels increases input size without adding new visual information.
- **Normalization mismatch:** ImageNet-pretrained models expect RGB channel statistics, which may not match medical scan intensity distributions.
- **Feature bias:** pretrained filters may emphasize natural-image color and texture patterns that are less relevant to medical imaging.
- **Pipeline instability:** inconsistent image loading through RGB/BGR/grayscale conversions can create shape mismatches or distorted inputs.

Dr. AI addresses this by using a grayscale-optimized image pipeline:

- Explicit grayscale standardization.
- Single-channel input shape: `(224, 224, 1)`.
- Custom ResNet50-style CNN models trained from scratch.
- Contrast enhancement with CLAHE.
- Separate task-specific models for brain and lung classification.
- Validation with 5-fold cross-validation and held-out test sets.

The core contribution is not the invention of a new CNN block, but the design and implementation of a medical-scan-aligned ResNet50 pipeline that avoids forcing grayscale MRI/CT data into RGB assumptions.

---

## System Architecture

Dr. AI follows a modular, multi-layered architecture:

```text
User Upload
    |
    |-- Medical Image
    |      |
    |      |-- Preprocessing
    |      |      - Grayscale loading
    |      |      - CLAHE contrast enhancement
    |      |      - Resize to 224x224
    |      |      - Normalization
    |      |
    |      |-- CNN Classification
    |      |      - Brain ResNet50 model
    |      |      - Lung ResNet50 model
    |      |
    |      |-- Grad-CAM++ Explainability
    |      |
    |      |-- Prediction Result Page
    |
    |-- Medical Report
           |
           |-- Text Extraction
           |      - PDF parsing
           |      - OCR fallback
           |      - Text file reading
           |
           |-- LLM Summarization
           |
           |-- Structured HTML Summary
           |
           |-- Downloadable PDF Report
```

The application backend is implemented in `app.py`, while OpenAI API interactions are encapsulated in `ChatGPTApi.py`.

---

## Machine Learning Models

The project contains two trained Keras models:

| Model | File | Task | Input Shape |
|---|---|---|---|
| Brain Tumor Classifier | `models/brain_resnet50.keras` | 4-class brain MRI classification | `(224, 224, 1)` |
| Lung Cancer Classifier | `models/lung_resnet50model.keras` | 4-class lung scan classification | `(224, 224, 1)` |

Both models are loaded at Flask startup using TensorFlow/Keras.

### Brain Tumor Classes

```python
[
    "Brain - No Brain Tumor",
    "Brain - Pituitary Brain Tumor",
    "Brain - Meningioma Brain Tumor",
    "Brain - Glioma Brain Tumor"
]
```

### Lung Cancer Classes

```python
[
    "Squamous Cell",
    "Adenocarcinomas Cell",
    "Normal cell",
    "Large cell"
]
```

---

## Datasets

The project report describes two main medical imaging datasets:

### Brain Tumor Dataset

- Source type: Kaggle brain MRI dataset.
- Approximate size: 14,000 images.
- Modality: MRI.
- Format: grayscale medical images.
- Classes:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor

### Lung Cancer Dataset

- Source type: IQ-OTHNCCD lung cancer dataset.
- Original size: approximately 9,000 images.
- Augmented size: 12,000+ images.
- Classes:
  - Squamous Cell Carcinoma
  - Adenocarcinoma
  - Large Cell Carcinoma
  - Normal Cell

Augmentation was used to increase training diversity and reduce class imbalance. Techniques included rotation, zoom, brightness and contrast adjustment, shear, and related geometric/intensity transformations.

---

## Algorithmic Approach

Dr. AI uses a supervised deep learning classification approach for scan analysis and a prompt-driven LLM workflow for report interpretation.

### Image Classification

The image classification module is based on a custom ResNet50-style convolutional neural network. ResNet-style residual connections are useful for deeper CNN training because they reduce vanishing-gradient issues and allow the model to learn residual mappings across convolutional blocks.

Unlike many standard ResNet50 pipelines, this project does not force grayscale medical scans into three-channel RGB format. Instead, the model accepts single-channel input directly:

```text
Input: 224 x 224 x 1
Output: class probabilities over 4 disease categories
```

This design better matches MRI/CT image characteristics and avoids unnecessary RGB channel duplication.

### Report Interpretation

The report analyzer uses a hybrid pipeline:

1. Extract text from medical documents.
2. Clean and pass extracted text into a structured prompt.
3. Ask the LLM to produce a user-readable HTML summary.
4. Render the result in the web interface.
5. Convert the summary into a downloadable PDF when requested.

This means the project combines classic document processing, OCR, prompt engineering, and web delivery.

---

## Training Methodology

The model training process was designed around robustness and generalization.

### Preprocessing

- Explicit grayscale conversion.
- Resizing to `224 x 224`.
- Pixel normalization to `[0, 1]`.
- CLAHE contrast enhancement.
- Augmentation for model robustness and class balancing.

### Optimization

The models were trained using:

- Adam optimizer.
- Categorical cross-entropy loss.
- Label smoothing to reduce overconfident predictions.
- Class weights to handle imbalance.
- Dropout and L2 regularization to reduce overfitting.
- Early stopping based on validation performance.
- Learning rate scheduling.

### Validation Strategy

The project uses a two-phase evaluation design:

1. **5-fold cross-validation**
   - The dataset is split into five folds.
   - Each fold is used once as validation data.
   - Metrics are averaged across folds.

2. **Held-out test evaluation**
   - After cross-validation, the best-performing configuration is retrained.
   - Final performance is measured on an unseen test set.

This gives both fold-level robustness estimates and final test-set performance.

---

## Image Processing Pipeline

The image preprocessing pipeline is designed for grayscale medical imaging.

Steps:

1. Read uploaded image in grayscale.
2. Apply CLAHE for local contrast enhancement.
3. Resize image to `224 x 224`.
4. Convert pixel values to `float32`.
5. Normalize pixel values to `[0, 1]`.
6. Expand dimensions to produce model-ready shape:

```text
(224, 224)
 -> (224, 224, 1)
 -> (1, 224, 224, 1)
```

This matches the single-channel model design and avoids RGB duplication.

---

## Grad-CAM Explainability

Dr. AI includes Grad-CAM++ visual explanations for image predictions.

The system:

1. Uses the final convolutional layer `conv5_block3_out`.
2. Computes class-specific gradients.
3. Generates a heatmap showing the regions most responsible for the prediction.
4. Overlays the heatmap on the original scan.
5. Saves the output under `static/gradcam_images`.

Grad-CAM helps users and reviewers understand whether the model is focusing on relevant anatomical or pathological regions rather than irrelevant background artifacts.

---

## Medical Report Analyzer

The report analyzer accepts:

- PDF reports
- Scanned image reports
- Prescription images
- Text files

Processing flow:

1. Uploaded report is saved.
2. If PDF:
   - Text extraction is attempted with `pdfminer.six`.
   - If no text is found, the PDF is converted to images and OCR is applied.
3. If image:
   - EasyOCR extracts readable text.
4. If text file:
   - Raw text is read directly.
5. Extracted text is passed to the LLM prompt.
6. The model returns a structured HTML summary.
7. The user can download the result as a PDF.

The generated summary includes sections such as:

- Patient Report Summary
- Urgent Medical Actions
- Key Health Conditions
- Medicines and Prescriptions
- Test Findings and Summary
- Lifestyle and Diet Recommendations
- Long-Term Health Strategy
- Additional Notes and Clarifications
- Specific Parameters and Medical Values
- General Advice and Assurance

---

## Chatbot Assistant

The homepage includes a floating chatbot interface. The chatbot sends user messages to the Flask `/chatbot` endpoint, which forwards the message to the OpenAI chat completion API through `ChatGPTApi.py`.

The chatbot is instructed to act as a medical assistant and politely decline unrelated non-medical questions.

---

## Evaluation and Results

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion matrices
- 5-fold cross-validation
- Held-out test set evaluation
- Grad-CAM visual inspection

### Cross-Validation Results

#### Brain Tumor Classification

| Metric | Average ± Std |
|---|---:|
| Accuracy | 91.01 ± 5.85 |
| Precision | 92.26 ± 4.04 |
| Recall | 91.01 ± 5.85 |
| F1-score | 90.87 ± 6.10 |

#### Lung Cancer Classification

| Metric | Average ± Std |
|---|---:|
| Accuracy | 91.77 ± 4.06 |
| Precision | 92.39 ± 3.56 |
| Recall | 91.77 ± 4.06 |
| F1-score | 91.77 ± 4.07 |

### Held-Out Test Set Results

| Task | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Lung Cancer Detection | 94.66% | 94.00% | 95.00% | 96.00% |
| Brain Tumor Classification | 96.18% | 97.00% | 96.00% | 96.25% |

### Per-Class Lung Cancer Results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Squamous Cell | 90.00% | 98.00% | 94.00% |
| Adenocarcinoma Cell | 97.00% | 92.00% | 94.00% |
| Normal | 99.00% | 100.00% | 99.00% |
| Large Cell | 95.00% | 89.00% | 92.00% |

### Per-Class Brain Tumor Results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| No Tumor | 92.00% | 100.00% | 96.00% |
| Pituitary | 99.00% | 98.00% | 98.00% |
| Meningioma | 98.00% | 88.00% | 93.00% |
| Glioma | 98.00% | 98.00% | 98.00% |

### Reported AUC-ROC

| Model | AUC-ROC |
|---|---:|
| Lung Disease Detection | 0.94 |
| Brain Tumor Classification | 0.96 |

---

## Technology Stack

### Backend

- Python 3.11
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- BeautifulSoup
- ReportLab

### Machine Learning

- Custom ResNet50-style CNNs
- Categorical cross-entropy
- Adam optimizer
- Label smoothing
- Class weights
- Dropout
- L2 regularization
- Early stopping
- Learning rate scheduling
- 5-fold cross-validation

### OCR and Text Processing

- `pdfminer.six`
- `pdf2image`
- `EasyOCR`
- OpenAI Chat Completions API

### Frontend

- HTML
- CSS
- JavaScript
- Bootstrap
- Jinja2 templates

### Visualization and Reporting

- Grad-CAM++
- OpenCV heatmap overlays
- ReportLab PDF generation

---

## Repository Structure

```text
Dr-AI app-final/
├── app.py                         # Main Flask backend
├── ChatGPTApi.py                  # OpenAI chat completion wrapper
├── requirements.txt               # Python dependencies
├── Dr_AI_Report.docx              # Research/project report
├── Brain_Tumor_resnet50.ipynb     # Brain model training notebook
├── Lung_cancer_resnet50.ipynb     # Lung model training notebook
├── models/
│   ├── brain_resnet50.keras       # Trained brain tumor classifier
│   └── lung_resnet50model.keras   # Trained lung cancer classifier
├── brain_live/                    # Example brain MRI images
├── lung_live/                     # Example lung scan images
├── Summarizer/                    # Example medical reports
├── static/
│   ├── assets/images/             # Uploaded and UI images
│   ├── gradcam_images/            # Generated Grad-CAM outputs
│   ├── main.js                    # Frontend interaction script
│   └── style.css                  # Main stylesheet
└── templates/
    ├── index.html                 # Homepage
    ├── header.html                # Navigation/header
    ├── upload.html                # Scan upload page
    ├── upload_report.html         # Report upload page
    ├── results.html               # Image prediction result page
    └── report_result.html         # Report analyzer result page
```

---

## Implementation Details

### Backend Entry Point

The main Flask app lives in `app.py`. It is responsible for:

- Loading trained Keras models.
- Handling file uploads.
- Routing users to scan upload and report upload pages.
- Running preprocessing and prediction.
- Generating Grad-CAM++ visualizations.
- Extracting text from reports.
- Sending report text to the LLM pipeline.
- Rendering prediction/report result pages.
- Generating downloadable PDF summaries.

### OpenAI API Wrapper

`ChatGPTApi.py` contains the chat-completion wrapper used by both the chatbot and report analyzer.

For production use, the API key should be loaded from an environment variable rather than stored in source code.

### Templates

The `templates/` folder contains the Jinja pages:

- `index.html`: homepage and chatbot interface.
- `upload.html`: medical scan upload UI.
- `upload_report.html`: report upload UI.
- `results.html`: model prediction and Grad-CAM result page.
- `report_result.html`: rendered report summary and PDF download form.
- `header.html`: shared navigation.

### Static Assets

The `static/` folder stores:

- UI images and logos.
- CSS and JavaScript.
- Uploaded scan/report images.
- Generated Grad-CAM overlays.
- Generated downloadable report PDFs.

### Generated Artifacts

During use, the application may generate:

- Uploaded files under `static/assets/images`.
- Grad-CAM overlays under `static/gradcam_images`.
- Report Analyzer PDFs under the configured upload folder.

---

## Limitations

Dr. AI is a research prototype and has important limitations:

- It has not been clinically certified.
- Model performance depends on dataset quality and similarity to real-world scans.
- Medical scan quality, angle, contrast, and acquisition protocol may affect prediction reliability.
- Grad-CAM visualizations support interpretability but do not prove clinical correctness.
- The report analyzer uses an LLM and may produce incomplete or incorrect summaries.
- OCR quality may degrade with handwritten, low-resolution, rotated, or noisy reports.
- The chatbot should not be used for emergency medical decision-making.

**Users should always consult qualified medical professionals for diagnosis, treatment, and urgent concerns.**

---

## Future Work

Planned or recommended improvements include:

- Fine-tuning a domain-specific medical summarization model.
- Adding multilingual report support.
- Improving OCR for handwritten prescriptions.
- Adding confidence calibration and uncertainty estimation.
- Adding segmentation support for tumor localization.
- Expanding to additional medical conditions.
- Adding clinician feedback loops.
- Testing on larger external datasets.
- Implementing stronger privacy and security controls.
- Deploying optimized models for edge or offline inference.
- Adding Docker support and CI/CD pipelines.

---

## Author

**Ayan Khan**

Project: **Dr. AI - Medical Image Diagnostic System for Automated Image and Text Analysis**

This project was developed as an AI-powered medical diagnostic assistant combining deep learning, explainability, OCR, and natural language processing to make medical information more accessible and understandable.

