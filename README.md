# Digit Recognition using MNIST Dataset

This project implements a digit recognition model using the MNIST dataset. The model is built using **TensorFlow** and **Keras** and uses a neural network to classify handwritten digits (0-9).

## Project Overview

This project aims to recognize handwritten digits by training a neural network on the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of digits.

## Dataset

The **MNIST** dataset is a widely used dataset of handwritten digits. It includes:
- **60,000 training images** and **10,000 test images**.
- Each image is a **28x28 grayscale image**, flattened into a 784-dimensional vector for input into the neural network.

## Model Architecture

The project uses a simple feed-forward neural network. The models are built using Keras `Sequential` API and trained using the **Adam optimizer** and **Sparse Categorical Crossentropy** loss.

There are two variations of the model in this project:

1. **Single-layer Dense Network**:
   - A single dense layer with 10 neurons (one for each class) and `sigmoid` activation.
   
2. **Two-layer Dense Network**:
   - The first layer is a dense layer with 100 neurons and `ReLU` activation.
   - The output layer is a dense layer with 10 neurons and `sigmoid` activation.

3. **Flattened Input Model**:
   - First, the image input is flattened to a 784-dimensional vector.
   - Then, the first layer is a dense layer with 100 neurons and `ReLU` activation.
   - Finally, the output layer is a dense layer with 10 neurons and `sigmoid` activation.

## Results

After training the models for 5 epochs, the models achieve high accuracy on the test set.

A **confusion matrix** is generated to visualize the performance of the model, showing where the model performs well and where it might misclassify digits.

## Installation

To run this project, ensure you have Python 3.x installed along with the necessary libraries:
- TensorFlow
- Keras
- Matplotlib
- Numpy
- Seaborn

You can install the required libraries using the following command:

```bash
pip install tensorflow matplotlib numpy seaborn
```

## How to Run

1. Load the dataset and preprocess the data.
2. Train the model using the MNIST dataset.
3. Evaluate the model using the test dataset.
4. Visualize predictions and the confusion matrix.

To run the code, you can either execute the Jupyter Notebook or run it as a Python script.

## Usage

1. **Load and preprocess data:**
   ```python
   (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
   ```

2. **Train the model:**
   ```python
   model = keras.Sequential([
       keras.layers.Dense(100, input_shape=(784,), activation='relu'),
       keras.layers.Dense(10, activation='sigmoid')
   ])
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(X_train_flattened, y_train, epochs=5)
   ```

3. **Evaluate the model:**
   ```python
   model.evaluate(X_test_flattened, y_test)
   ```

4. **Generate Confusion Matrix:**
   ```python
   cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
   sn.heatmap(cm, annot=True, fmt='d')
   ```

## Future Improvements

- Implement a more advanced model using **Convolutional Neural Networks (CNN)** to improve accuracy.
- Experiment with **different optimizers** and **learning rates** for better performance.
- Use **data augmentation** to improve the generalization of the model.
