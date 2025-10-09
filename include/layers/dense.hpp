#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>

/**
 * @brief A fully connected (dense) layer for a neural network.
 */
struct Dense {
    Eigen::MatrixXf weights;     // The weight matrix of the layer.
    Eigen::VectorXf biases;      // The bias vector of the layer.
    float learning_rate;         // The learning rate for parameter updates.

    Eigen::VectorXf cached_input; 

    /**
     * @brief Construct a new Dense layer.
     */
    Dense(int input_dimension, int output_dimension, float lr = 0.01f)
        : weights(output_dimension, input_dimension),
          biases(Eigen::VectorXf::Zero(output_dimension)),
          learning_rate(lr) {
        
        std::mt19937 rng(123); 
        float scale = std::sqrt(2.0f / input_dimension);
        std::normal_distribution<float> normal_dist(0.0f, scale);

        for (int row = 0; row < weights.rows(); ++row) {
            for (int col = 0; col < weights.cols(); ++col) {
                weights(row, col) = normal_dist(rng);
            }
        }
    }

    /**
     * @brief Performs the forward pass of the dense layer.
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& input_vector) {
        cached_input = input_vector; 
        return weights * input_vector + biases;
    }

    /**
     * @brief Performs the backward pass (backpropagation) and updates parameters.
     */
    Eigen::VectorXf backward(const Eigen::VectorXf& output_gradient) {
        Eigen::MatrixXf weights_gradient = output_gradient * cached_input.transpose();
        Eigen::VectorXf biases_gradient = output_gradient;
        
        // Gradient for the input is the weight matrix (transposed) multiplied by the output gradient.
        Eigen::VectorXf input_gradient = weights.transpose() * output_gradient;

        // --- Update parameters using gradient descent --- //
        weights -= learning_rate * weights_gradient;
        biases -= learning_rate * biases_gradient;

        return input_gradient;
    }
};