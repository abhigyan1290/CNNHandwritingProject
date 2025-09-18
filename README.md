# MNIST Convolutional Neural Network (C++ / Eigen)

This project trains a small Convolutional Neural Network (CNN) from scratch in C++ using the [Eigen](https://eigen.tuxfamily.org/) linear-algebra library (and optional OpenCV for quick visual checks).  
It classifies handwritten digits from the MNIST dataset.

## 🚀 Overview of the Pipeline
The training executable (`mnist_cnn`) performs these stages:

### 1️⃣ Load the MNIST data
* **What**: Read the four raw IDX files (`train-images`, `train-labels`, `t10k-images`, `t10k-labels`) into memory as Eigen `MatrixXf` objects.  
* **Why**: Converts the dataset into normalized floating-point tensors that the network can consume.  
* **Key detail**: Each image is scaled to `[0,1]` so learning is stable.

### 2️⃣ Define the CNN model
Architecture:
```
Conv(1→8, k=3, same) → ReLU → MaxPool 2×2
→ Conv(8→16, k=3, same) → ReLU → MaxPool 2×2
→ Flatten → Dense 128 → ReLU → Dense 10 → Softmax
```
* **Why this design**  
  * Convolutions extract spatial features (edges, strokes).  
  * ReLU adds non-linearity and keeps gradients healthy.  
  * MaxPool reduces spatial size and adds translation invariance.  
  * Dense layers combine features into class scores.  
  * Softmax converts scores to probabilities for cross-entropy loss.

### 3️⃣ Forward pass
* **What**: For each training image, propagate activations through every layer.
  * Conv layers slide learnable filters to detect patterns.
  * ReLU applies `max(0,x)` to introduce non-linearity.
  * MaxPool downsamples by taking the maximum in each 2×2 window.
  * Flatten converts the 3-D feature map to a 1-D vector.
  * Dense layers compute `Wx + b` to produce final logits.
* **Why**: Produces predictions (`logits`) and intermediate activations needed for backprop.

### 4️⃣ Compute loss
* **What**: Apply **softmax cross-entropy**  
  `L = −log( exp(logit_y) / Σ exp(logits) )`
* **Why**: Cross-entropy measures how well predicted probabilities match the true digit; it gives well-behaved gradients for classification.

### 5️⃣ Backward pass (backpropagation)
* **What**: Propagate the gradient of the loss back through each layer to compute:
  * dL/dWeights and dL/dBias for all layers.
  * dL/dInput for convolution layers (used by previous layer).
* **Why**: Tells each parameter how to move to reduce the loss.

### 6️⃣ Parameter update
* **What**: **Stochastic Gradient Descent**  
  `W := W − learning_rate * dW`
* **Why**: Incrementally nudges weights in the direction that lowers the loss.

### 7️⃣ Training loop
* **What**: Repeat forward → loss → backward → update over many random samples (epochs).  
  Periodically evaluate accuracy on a test subset and print progress.
* **Why**: Iterative optimization gradually minimizes the loss and improves classification accuracy.

### 8️⃣ Evaluation
* After each epoch, run a forward pass on the test set and compute accuracy:
  * `accuracy = correct_predictions / total_samples`
* **Why**: Monitors generalization to unseen data.

## ⚡️Performance Tips
* **Memoized operations**: Eigen’s expression templates keep matrix math fast.
* **Same padding** in convolutions preserves spatial size, simplifying pooling.
* **He initialization** keeps signal variance stable when using ReLU.

## 🔧 Build & Run
```bash
mkdir build && cd build
cmake ..
cmake --build . -j
./mnist_cnn \\
  path/to/train-images.idx3-ubyte \\
  path/to/train-labels.idx1-ubyte \\
  path/to/t10k-images.idx3-ubyte \\
  path/to/t10k-labels.idx1-ubyte
```

## 📚 Why build this from scratch?
Implementing each layer yourself forces you to understand:
* The exact math of convolutions, pooling, and backprop.
* How memory layout and numerical stability affect training.
* How deep-learning frameworks (PyTorch, TensorFlow) work under the hood.

This makes you far more confident when debugging or optimizing real production models.

**End result:**  
After a few epochs the model reaches **>95 % test accuracy**—and you can explain every step, from bytes on disk to gradients in memory.
