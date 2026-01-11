# Dataset, Preprocessing, and Annotations

## 📂 Raw, Processed, and Annotated Data (External Storage)

Due to GitHub storage limitations, large datasets used in this project are hosted externally on Google Drive.

🔗 **Access the dataset here:**  
👉 [Google Drive – Dataset and Annotations](https://drive.google.com/drive/folders/1qzeJfM_mQHZTWfg0zQPu6hOE0ighwh3t?usp=sharing)

The dataset includes:
- Raw ultrasound videos
- Extracted and preprocessed frames
- Annotated labels (YOLOv8 segmentation format)
- Supporting metadata files

> Note: All data is anonymized and provided strictly for research and demonstration purposes.

This document describes the raw data, preprocessing steps, and annotation pipeline used in this project.

---

## Raw Data
- Source: Transvaginal ultrasound (TVS) videos
- Format: Clinical ultrasound video recordings
- Content: Ovarian scans for follicular monitoring

---

## Data Organization
- Raw videos stored separately due to size constraints
- Large files are hosted externally (Google Drive)
- Repository contains sample data and metadata only

---

## Frame Extraction
- Videos converted into individual frames
- Frame sampling rate optimized to reduce redundancy
- Ensures coverage across follicular development stages

---

## Image Preprocessing
- Noise reduction
- Contrast enhancement
- Normalization for model compatibility
- Resizing to match YOLOv8 input requirements

---

## Annotation Pipeline
- Platform: **Roboflow**
- Annotation Type: Polygon-based instance segmentation
- Annotation Target:
  - Individual ovarian follicles
- Label Format:
  - YOLOv8 segmentation format

---

## Annotated Dataset Structure
- Images folder:
  - Preprocessed ultrasound frames
- Labels folder:
  - Corresponding segmentation masks and annotations

---

## Data Volume (Approximate)
- Ultrasound videos: Multiple sessions
- Extracted frames: Large-scale (hosted externally)
- Annotated images: Curated subset for training

---

## Data Usage
- Training YOLOv8-S segmentation model
- Validation and testing
- Streamlit app inference

---

## Data Limitations
- Annotation quality depends on ultrasound clarity
- Limited availability of labeled medical data

---

## Ethical Considerations
- No personally identifiable patient information included
- Data used strictly for research and educational purposes

---

## Future Dataset Expansion
- More diverse ultrasound sources
- Multi-center clinical data
- Additional annotation classes
