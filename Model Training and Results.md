# Model Training, Fine-Tuning, and Results
---

## 📂 Model Training Outputs & Results (External Storage)

Due to GitHub storage limitations, large training artifacts such as:
- Training logs
- Validation outputs
- Segmentation result images
- Model performance visualizations
- Experiment outputs

are hosted externally on Google Drive.

🔗 **Click here to access Model Training and Results**  
👉 [Google Drive – Model Training Outputs](https://drive.google.com/drive/folders/1csZTUybdpujH5j-JiCOjJVjFyhsWtwwu?usp=sharing)

---

This document describes the training process, fine-tuning strategy, and performance results of the ovarian follicle detection and segmentation model used in this project.

---

## Model Architecture
- **Model:** YOLOv8-S (Instance Segmentation)
- **Framework:** Ultralytics YOLOv8
- **Task:** Instance segmentation of ovarian follicles in ultrasound frames

---

## Dataset Overview
- Source: Transvaginal ultrasound (TVS) videos
- Frames extracted from videos at controlled intervals
- Annotated using polygon-based instance segmentation

---

## Annotation Details
- Tool: **Roboflow (Free Polygon Annotation Tool)**
- Annotation Type: Instance segmentation masks
- Classes:
  - Ovarian Follicle

---

## Training Configuration
- Input Image Size: 640 × 640
- Batch Size: Tuned based on GPU memory
- Optimizer: Default YOLOv8 optimizer
- Loss Components:
  - Segmentation loss
  - Bounding box regression loss
  - Classification loss

---

## Fine-Tuning Strategy
- Pretrained YOLOv8-S weights used as initialization
- Fine-tuned on domain-specific ultrasound data
- Learning rate scheduling applied to stabilize convergence
- Overfitting monitored using validation metrics

---

## Model Outputs
- Trained model file: `best.pt`
- Outputs include:
  - Segmentation masks
  - Bounding boxes
  - Confidence scores per detected follicle

---

## Performance Summary
- Accurate localization of follicles in noisy ultrasound images
- Reliable size estimation from segmentation masks
- Effective generalization across varying ultrasound quality levels

---

## Clinical Relevance
- Follicles with diameter **≥ 18 mm** are classified as **mature**
- Smaller follicles are classified as **immature**
- Model outputs support IVF follicular monitoring and decision-making

---

## Limitations
- Model performance may vary with extreme image noise
- Further calibration required for confidence score interpretability

---

## Future Improvements
- Confidence calibration using temperature scaling
- Multi-class follicle grading
- Larger annotated dataset for improved robustness
