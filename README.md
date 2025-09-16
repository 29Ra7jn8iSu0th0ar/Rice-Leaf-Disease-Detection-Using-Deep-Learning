# Rice Leaf Disease Detection Using Deep Learning

## Project Overview
This project aims to develop a system for detecting and classifying diseases in rice leaves using machine learning and deep learning techniques. Early and accurate detection is crucial for preventing crop loss and ensuring food security. The system utilizes four different models to maximize classification accuracy.

---

## Dataset
- **Source:** Images of rice leaves affected by various diseases.
- **Complexity:** Images include varied backgrounds, making detection challenging.
- **Preprocessing:** Images were resized, normalized, and outliers removed to improve data quality.

---

## Models Used

| Model                  | Description                                                                        | Accuracy |
|------------------------|------------------------------------------------------------------------------------|----------|
| **MobileNet V2**       | Lightweight model for mobile/embedded vision applications.                         | 92%      |
| **EfficientNet B0**    | Efficient CNN that balances accuracy and computation.                              | 94%      |
| **Inception V3**       | Deep CNN known for high accuracy and computational efficiency.                     | 91%      |
| **SVM + ORB Features** | Classical ML model using ORB (Oriented FAST and Rotated BRIEF) for image features. | 88%      |

---

## Implementation Details

1. **Data Preprocessing**
   - Image resizing and normalization
   - Data augmentation to increase training diversity
   - Splitting into training and validation sets

2. **Feature Extraction**
   - ORB (Oriented FAST and Rotated BRIEF) features extracted for SVM model

3. **Model Training**
   - Deep learning models (MobileNet V2, EfficientNet B0, Inception V3) trained using transfer learning
   - SVM trained on ORB features

4. **Model Evaluation**
   - Evaluated using accuracy, precision, recall, and F1-score
   - Performed cross-validation for robustness

---

## Results

- All models were compared, and the best performer was selected based on both accuracy and generalization to unseen data.
- **EfficientNet B0** achieved the highest accuracy at **94%**.

---

## Conclusion

This project demonstrates how machine learning and deep learning can be applied to address agricultural challenges. By employing multiple models and extensive data preprocessing, the system provides accurate and reliable detection of rice leaf diseases. This can help farmers take timely action, contributing to improved crop management and food security.

---
