import numpy as np
np.random.seed(420)

# --- Activation Functions ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Extracted from:
# https://github.com/xbeat/Machine-Learning/blob/main/Building%20a%20Softmax%20Activation%20Function%20from%20Scratch%20in%20Python.md
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# --- Loss Functions ---
def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss, axis=0)

# --- Accuracy Calculation ---
def calculate_accuracy(X, y_true):
    correct_predictions = 0
    for i in range(X.shape[0]):
        hidden_layer_1_output = relu(np.dot(X[i], weights['W1']) + biases['b1'])
        hidden_layer_2_output = relu(np.dot(hidden_layer_1_output, weights['W2']) + biases['b2'])
        predicted_output = softmax(np.dot(hidden_layer_2_output, weights['W3']) + biases['b3'])

        predicted_class = np.argmax(predicted_output)
        true_class = np.argmax(y_true[i])

        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / X.shape[0]
    return accuracy

# --- Load Data ---
fname = 'assign1_data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)

# One hot encoding
# Extracted from:
# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
num_classes = np.max(y) + 1
y_one_hot = np.zeros((y.size, num_classes))
y_one_hot[np.arange(y.size), y] = 1

# Train-Test split
X_train, y_train = X[:400], y_one_hot[:400]
X_test, y_test = X[400:], y_one_hot[400:]

# --- Network Initialization ---
nn_architecture = [
    {"input_dim": 3, "output_dim": 4},
    {"input_dim": 4, "output_dim": 8},
    {"input_dim": 8, "output_dim": 3}
]

weights = {}
biases = {}

for i, layer in enumerate(nn_architecture):
    layer_i = i + 1
    layer_input_size = layer["input_dim"]
    layer_output_size = layer["output_dim"]
    weights['W%d' % layer_i] = np.random.randn(layer_input_size, layer_output_size) * 0.1
    biases['b%d' % layer_i] = np.zeros((1, layer_output_size))

# Training parameters
learning_rate = 0.1
epochs = 9
batch_size = 10 

# --- Training ---
for epoch in range(epochs):
    total_loss = 0
    indices = np.arange(X_train.shape[0])
    
    # Shuffle the data at the start of each epoch
    np.random.shuffle(indices)  
    
    for start_idx in range(0, X_train.shape[0], batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        # Initialize accumulators for weight and bias gradients
        dW1_accum = np.zeros_like(weights['W1'])
        dW2_accum = np.zeros_like(weights['W2'])
        dW3_accum = np.zeros_like(weights['W3'])
        db1_accum = np.zeros_like(biases['b1'])
        db2_accum = np.zeros_like(biases['b2'])
        db3_accum = np.zeros_like(biases['b3'])

        for i in range(X_batch.shape[0]):
            # Forward pass
            hidden_layer_1_input = np.dot(X_batch[i], weights['W1']) + biases['b1']
            hidden_layer_1_output = relu(hidden_layer_1_input)

            hidden_layer_2_input = np.dot(hidden_layer_1_output, weights['W2']) + biases['b2']
            hidden_layer_2_output = relu(hidden_layer_2_input)
            output_layer_input = np.dot(hidden_layer_2_output, weights['W3']) + biases['b3']
            predicted_output = softmax(output_layer_input)

            # Compute loss
            total_loss += cross_entropy_loss(y_batch[i:i+1], predicted_output)

            # Backpropagation
            error = predicted_output - y_batch[i]
            d_predicted_output = error
            
            hidden_layer_2_error = d_predicted_output.dot(weights['W3'].T)
            d_hidden_layer_2_output = hidden_layer_2_error * relu_derivative(hidden_layer_2_output)

            hidden_layer_1_error = d_hidden_layer_2_output.dot(weights['W2'].T)
            d_hidden_layer_1_output = hidden_layer_1_error * relu_derivative(hidden_layer_1_output)

            # Accumulate gradients for this mini-batch
            dW3_accum += hidden_layer_2_output.reshape(-1, 1).dot(d_predicted_output.reshape(1, -1))
            dW2_accum += hidden_layer_1_output.reshape(-1, 1).dot(d_hidden_layer_2_output.reshape(1, -1))
            dW1_accum += X_batch[i].reshape(-1, 1).dot(d_hidden_layer_1_output.reshape(1, -1))
            db3_accum += d_predicted_output
            db2_accum += d_hidden_layer_2_output
            db1_accum += d_hidden_layer_1_output

        # Update weights and biases after each mini-batch
        weights['W3'] -= dW3_accum * learning_rate / batch_size
        weights['W2'] -= dW2_accum * learning_rate / batch_size
        weights['W1'] -= dW1_accum * learning_rate / batch_size
        biases['b3'] -= db3_accum * learning_rate / batch_size
        biases['b2'] -= db2_accum * learning_rate / batch_size
        biases['b1'] -= db1_accum * learning_rate / batch_size

    # Print the average loss for the epoch
    print(f"Epoch {epoch}, Loss: {total_loss / X_train.shape[0]}")

    # Calculate and print accuracy on training set
    train_accuracy = calculate_accuracy(X_train, y_train)
    print(f"Epoch {epoch}, Training Accuracy: {train_accuracy * 100:.2f}%")

# After the loop, calculate test accuracy
accuracy = calculate_accuracy(X_test, y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
