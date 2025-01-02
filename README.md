# Screening Test: Practical Task


## Objective: Demonstrate your ability to work with neural networks and data.



## Task Steps:

1. Download the MNIST dataset.
2. Build a simple neural network to classify handwritten digits (use TensorFlow).
3. Perform the following:
4. Preprocess the data (normalize and split into train/validation/test).
5. Train the model for at least 5 epochs.
6. Evaluate the model and report accuracy.
7. Save the trained model and share the code via GitHub or a similar platform.


## Deliverables:

Provide your code, instructions to run it, and a brief summary of your approach and results (accuracy, insights, etc.).

## Instructions to run the code
1. pip install -r requirements.txt 
    - This step will install the required dependencies for the project (tensorflow and sci-kit learn)
2. Navigate to the folder with io_health.py
3. Run the following command:
    - python io_health.py

## Summary of Approach and Results

### Approach

1. Data Preprocessing
    - Loaded the MNIST dataset using TensorFlow's keras.datasets.
    - Normalized the pixel values to the range [0, 1] using MinMaxScaler from Scikit-learn for consistent scaling.
    - Split the training data into training and validation sets (80%-20%) using train_test_split with a fixed random_state=77 for reproducibility.
2. Model Architecture
    - Built a feedforward neural network using TensorFlow:
    - Flatten layer to convert 2D images into a 1D array.
    - Hidden layer with 128 units and ReLU activation.
    - Dropout layer with a 20% rate to prevent overfitting.
    - Output layer with 10 units and a softmax activation for multi-class classification.
2. Training
    - Used the Adam optimizer and sparse categorical crossentropy as the loss function.
    - Trained the model for 5 epochs with validation on a separate validation set.
3. Evaluation
    - Evaluated the model on the test set to measure its generalization performance.
4. Model Saving
    - Saved the trained model as mnist_model.h5 for future use.
    
### Results
*Accuracy* : Around 97.40% after 5 epochs, demonstrating excellent learning on unseen validation data.

* Note: The .h5 file is bigger than the file limit on git so it will be generated upon running the io_health.py file.