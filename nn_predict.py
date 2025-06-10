import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum (0,x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)



# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
    import numpy as np

# Assuming you have:
# X_test: test images (NumPy array)
# y_test: test labels (NumPy array)
# model_forward: function that does forward pass and returns probabilities

# 1. Run forward pass (adjust according to your code)
y_pred_probs = model_forward(X_test, model_architecture, model_weights)  # shape (num_samples, 10)

# 2. Get predicted labels
y_pred = np.argmax(y_pred_probs, axis=1)

# 3. Calculate accuracy
accuracy = np.mean(y_pred == y_test)

# 4. Write accuracy to test_acc.txt
with open('test_acc.txt', 'w') as f:
    f.write(f"{accuracy}\n")

print(f"Test Accuracy saved to test_acc.txt: {accuracy:.4f}")

    
