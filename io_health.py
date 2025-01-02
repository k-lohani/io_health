import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load and preprocess data
def load_and_preprocess_data(random_state=42):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    
    # Split training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=random_state
    )
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# Step 2: Build the neural network model
def build_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Step 3: Compile, train, and evaluate the model
def train_and_evaluate_model(model, data):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_accuracy}")
    return test_accuracy

# Step 4: Save the trained model
def save_model(model, filename="mnist_model.h5"):
    model.save(filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    random_state = 77
    data = load_and_preprocess_data(random_state=random_state)
    model = build_model()
    accuracy = train_and_evaluate_model(model, data)
    save_model(model)
