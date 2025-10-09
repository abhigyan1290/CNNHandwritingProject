#pragma once

#include <Eigen/Dense>
#include <vector>

// Include all the refactored layer definitions.
#include "layers/conv2d.hpp"
#include "layers/relu.hpp"
#include "layers/maxpool2x2.hpp"
#include "layers/flatten.hpp"
#include "layers/dense.hpp"
#include "layers/softmax.hpp"

/**
 * @brief A simple Convolutional Neural Network (CNN) for MNIST classification.
 *
 * This struct assembles various layers into a complete network architecture
 * inspired by LeNet. It defines the forward pass for inference, a training
 * step that includes backpropagation, and a prediction method.
 *
 * The architecture is as follows:
 * [Input: 28x28x1] -> Conv1(8 filters) -> ReLU -> MaxPool ->
 * Conv2(16 filters) -> ReLU -> MaxPool -> Flatten ->
 * Dense(128) -> ReLU -> Dense(10) -> [Output: Logits]
 */
struct CNN {
    //--- Layer Definitions ---//

    // First convolutional block
    Conv2D conv1;
    ReLU2D relu1;
    MaxPool2x2 pool1;

    // Second convolutional block
    Conv2D conv2;
    ReLU2D relu2;
    MaxPool2x2 pool2;

    // Classifier (fully connected head)
    Flatten flatten;
    Dense fc1;
    ReLU1D relu3;
    Dense fc2;

    // Combined Softmax and Cross-Entropy loss function
    SoftmaxCE loss_function;

    /**
     * @brief Constructs the CNN and initializes its layers.
     */
    CNN(float learning_rate) :
        // Conv1: 1 input channel (grayscale), 8 output channels, 3x3 kernel
        conv1(1, 8, 3, learning_rate),
        // Conv2: 8 input channels, 16 output channels, 3x3 kernel
        conv2(8, 16, 3, learning_rate),
        // FC1: Input size is calculated from the output of the conv blocks.
        // An initial 28x28 image becomes 14x14 after pool1, then 7x7 after pool2.
        // With 16 channels, the flattened size is 16 * 7 * 7. Output size is 128.
        fc1(16 * 7 * 7, 128, learning_rate),
        // FC2: 128 input features, 10 output features (for digits 0-9).
        fc2(128, 10, learning_rate) {}

    /**
     * @brief Performs a full forward pass through the network.
     */
    Eigen::VectorXf forward(const Eigen::MatrixXf& image) {
        std::vector<Eigen::MatrixXf> x = {image};

        // First convolutional block
        x = conv1.forward(x);
        x = relu1.forward(x);
        x = pool1.forward(x);

        // Second convolutional block
        x = conv2.forward(x);
        x = relu2.forward(x);
        x = pool2.forward(x);

        // Classifier head
        Eigen::VectorXf vec = flatten.forward(x);
        vec = fc1.forward(vec);
        vec = relu3.forward(vec);
        vec = fc2.forward(vec);

        return vec; // Return the final logits
    }

    /**
     * @brief Performs a single training step (forward and backward pass).
     */
    float train_step(const Eigen::MatrixXf& image, int label) {
        // --- Forward Pass ---
        Eigen::VectorXf logits = forward(image);
        float loss = loss_function.forward(logits, label);

        // The gradient flows backward from the loss function through each layer.
        auto gradient = loss_function.backward();
        gradient = fc2.backward(gradient);
        gradient = relu3.backward(gradient);
        gradient = fc1.backward(gradient);
        
        auto tensor_gradient = flatten.backward(gradient);
        tensor_gradient = pool2.backward(tensor_gradient);
        tensor_gradient = relu2.backward(tensor_gradient);
        tensor_gradient = conv2.backward(tensor_gradient);
        tensor_gradient = pool1.backward(tensor_gradient);
        tensor_gradient = relu1.backward(tensor_gradient);
        
        // The final gradient is passed to the first layer, completing the chain.
        (void)conv1.backward(tensor_gradient);

        return loss;
    }

    /**
     * @brief Predicts the class label for a given image.
     */
    int predict(const Eigen::MatrixXf& image) {
        Eigen::VectorXf logits = forward(image);
        
        // Find the index of the maximum logit, which corresponds to the predicted class.
        Eigen::Index predicted_index;
        logits.maxCoeff(&predicted_index);
        
        return static_cast<int>(predicted_index);
    }
};