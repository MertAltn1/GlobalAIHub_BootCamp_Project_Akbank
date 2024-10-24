# Fish Species Classification using Deep Learning

This project focuses on building a deep learning model using Convolutional Neural Networks (CNN) to classify different species of fish. The dataset used is sourced from Kaggle, containing multiple fish species in image format. The project involves data preprocessing, model development, training, evaluation, and optimization to achieve high classification accuracy.

## Dataset
The dataset consists of `.png` images of various fish species. The dataset can be accessed via [Kaggle Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset). This project uses image classification techniques to predict the fish species based on the given image data.

## Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **OpenCV**

## Project Steps

### 1. Data Loading and Preprocessing
- The fish images are loaded from the dataset, and the paths and labels are extracted.
- The data is stored in a Pandas DataFrame for easy manipulation and exploration.
- Images are rescaled, resized, and divided into training, validation, and test sets.

### 2. Data Augmentation
To improve the model's generalization ability, data augmentation techniques are applied:
- **Rescaling**
- **Rotation**
- **Zoom**

### 3. Model Development
A **Convolutional Neural Network (CNN)** is developed using Keras. The architecture includes:
- Multiple **convolutional layers** for feature extraction.
- **Pooling layers** to reduce spatial dimensions.
- **Fully connected layers** for classification.
- A **softmax** output layer to classify fish species into distinct categories.

### 4. Model Training
The model is trained on the training data with the following features:
- **Early stopping** is applied to prevent overfitting.
- **Exponential decay** learning rate is used for optimization.
- Validation data is used to monitor the training process.

### 5. Model Evaluation
After training, the model's performance is evaluated using:
- **Accuracy** on the test dataset.
- **Confusion matrix** for detailed class-based evaluation.
- **Classification report** showing precision, recall, and F1-score.

### 6. Hyperparameter Optimization
The model's performance is enhanced by fine-tuning various hyperparameters:
- **Number of layers**
- **Number of neurons in dense layers**
- **Dropout rate**
- **Learning rate**

Techniques like **EarlyStopping** and **learning rate scheduling** are employed to further improve performance.

## Results

The model achieved:
- **Training Accuracy**: 98.7%
- **Test Accuracy**: 99.6%

The model performed well on the dataset, with detailed performance metrics available in the confusion matrix and classification report.

## Kaggle Notebook Link
You can view and run the notebook on Kaggle via this link: [Kaggle Notebook](https://www.kaggle.com/code/musakaanaltin/kaan-mert-akbank-derin-ogrenme)

