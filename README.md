# Rice-Leaf-Disease-Detection-Using-Deep-Learning
**Project Overview**
This project aims to develop a system for detecting and classifying diseases in rice leaves using machine learning and deep learning techniques. Early and accurate detection of diseases in rice crops is crucial for preventing significant crop loss and ensuring food security. The project utilizes four different models to achieve high accuracy in disease classification.

**Dataset**
The dataset used in this project consists of images of rice leaves affected by various diseases. The images have varied backgrounds, making the detection task challenging. The dataset was preprocessed, normalized, and outliers were removed to enhance the quality of the data.

**Models Used**
**MobileNet V2**
Description: A lightweight deep learning model designed for mobile and embedded vision applications.
Accuracy Achieved: 92%

**EfficientNet B0**
Description: A highly efficient convolutional neural network model that balances accuracy and computational efficiency.
Accuracy Achieved: 94%

**Inception V3**
Description: A deep convolutional neural network that is known for its high accuracy and efficient use of computational resources.
Accuracy Achieved: 91%

**Support Vector Machine (SVM) with ORB features**
Description: A classical machine learning model using ORB (Oriented FAST and Rotated BRIEF) features for image classification.
Accuracy Achieved: 88%

**Implementation Details**
The implementation of this project involved the following steps:

**Data Preprocessing:**
Image resizing and normalization.
Data augmentation to increase the diversity of the training data.
Splitting the data into training and validation sets.
Feature Extraction:

For the SVM model, ORB (Oriented FAST and Rotated BRIEF) features were extracted from the images.
Model Training:

Each of the deep learning models (MobileNet V2, EfficientNet B0, Inception V3) was trained using transfer learning.
The SVM model was trained using the extracted ORB features.
Model Evaluation:

The models were evaluated using metrics such as accuracy, precision, recall, and F1-score.
Cross-validation was performed to ensure the robustness of the models.

**Results**
The performance of the models was compared, and the best-performing model was selected based on its accuracy and ability to generalize well to unseen data.

**Conclusion**
This project demonstrates the application of machine learning and deep learning techniques to agricultural problems. By employing multiple models and extensive data preprocessing, the project achieved high accuracy in detecting and classifying rice leaf diseases, providing valuable insights for farmers and contributing to better crop management practices.

