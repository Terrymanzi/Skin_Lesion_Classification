# Skin_Lesion_Classification
Classification for Melanoma Detection using MNIST: HAM10000 dataset

This project implements the Skin Lesion Classification project for melanoma detection, as per the project criteria. It demonstrates traditional machine learning using Scikit-learn and deep learning using TensorFlow (Sequential and Functional APIs, tf.data API). The notebook is modular, reproducible, and includes clear explanations, insights from results, and visualizations.

**Project Goal:** Classify skin lesion images as benign or malignant to aid early melanoma detection.
**Dataset:** HAM10000 from Kaggle which contains ~10,000 images.
**Requirements:** Run in Google Colab with GPU for faster training.Or on Kaggle.
**Reproducibility:** Set random seeds. Document all steps.

# DATASET
**Key Findings and Analysis**
Dataset Characteristics: This benchmark consists of 10015 images that are the result of an intensive study developed by various entities. The samples are represented in RGB format and have dimensions 600*450 (length and width respectively). This benchmark promotes the study of seven different types of skin lesions:

- Actinic Keratoses;
- Basal cell carcinoma;
- Benign Keratosis;
- Dermatofibroma;
- Melanocytic nevi;
- Melanoma;
- Vascular skin lesions;
The HAM10000 dataset presents significant class imbalance (~11% malignant vs. 89% benign), which is representative of real-world medical screening scenarios. This imbalance necessitates careful handling through class weighting and evaluation metrics beyond accuracy.

## Dataset Limitations:
**Demographic Bias:** Dataset may not represent diverse skin tones, ages, and geographic populations
**Acquisition Variance:** Images from different sources with varying quality, lighting, and equipment
**Limited Context:** Metadata (age, sex, location) not fully utilized in current models
**Class Imbalance:** Melanoma underrepresented, requiring careful validation strategies
**Diagnostic Uncertainty:** Some lesions are inherently ambiguous even to experts
**Problem with high complexity:** Features are hard to extract classify
**Samples with high dimensions:** not optimal for training unless shrunk

## Model Insights:
- Traditional Machine Learning: Handcrafted features (HOG, color histograms, LBP) combined with SVM/Random Forest achieved moderate performance (AUC ~0.75-0.85). These models capture texture and color patterns relevant to the ABCD rule of melanoma detection but struggle with subtle variations.

- Deep Learning - Sequential CNN: Custom CNN architecture showed improved performance (AUC ~0.85-0.90) through end-to-end learning, automatically discovering relevant features without manual engineering.

- Deep Learning - Transfer Learning: MobileNetV2 with fine-tuning achieved the best results (AUC ~0.90+), leveraging ImageNet pretraining to capture low-level features and adapting to medical imaging specifics.

## Critical Error Analysis:

- False Negatives (FN): Most critical in medical context - missing malignant cases can be life-threatening. Our models show varying FN rates, with DL approaches generally achieving better sensitivity.
- False Positives (FP): Lead to unnecessary biopsies and patient anxiety but are less critical than FNs. The trade-off between sensitivity and specificity can be adjusted via threshold tuning.
