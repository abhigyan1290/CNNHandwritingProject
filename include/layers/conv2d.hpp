#pragma once

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>

struct Conv2D {
    using Tensor = std::vector<Eigen::MatrixXf>;

    int input_channels;
    int output_channels;
    int kernel_size;
    float learning_rate;

    std::vector<Tensor> weights; 
    Eigen::VectorXf biases;      

    Tensor cached_input; 

    /**
     * @brief Construct a new Conv2D layer.
     */
    Conv2D(int in_channels, int out_channels, int k_size, float lr = 0.01f)
        : input_channels(in_channels),
          output_channels(out_channels),
          kernel_size(k_size),
          learning_rate(lr),
          biases(Eigen::VectorXf::Zero(out_channels)) {
        
        // Initialize weights using He initialization for better training dynamics.
        std::mt19937 rng(42); 
        float scale = std::sqrt(2.0f / (in_channels * k_size * k_size));
        std::normal_distribution<float> normal_dist(0.0f, scale);

        weights.resize(output_channels, Tensor(input_channels, Eigen::MatrixXf::Zero(k_size, k_size)));
        
        for (int out_idx = 0; out_idx < output_channels; ++out_idx) {
            for (int in_idx = 0; in_idx < input_channels; ++in_idx) {
                for (int row = 0; row < k_size; ++row) {
                    for (int col = 0; col < k_size; ++col) {
                        weights[out_idx][in_idx](row, col) = normal_dist(rng);
                    }
                }
            }
        }
    }

    /**
     * @brief Pads a matrix with zeros to maintain dimensions during convolution ("same" padding).
     */
    static Eigen::MatrixXf pad_with_zeros(const Eigen::MatrixXf& input_matrix, int pad_amount) {
        Eigen::MatrixXf padded_matrix = Eigen::MatrixXf::Zero(
            input_matrix.rows() + 2 * pad_amount,
            input_matrix.cols() + 2 * pad_amount
        );
        padded_matrix.block(pad_amount, pad_amount, input_matrix.rows(), input_matrix.cols()) = input_matrix;
        return padded_matrix;
    }

    /**
     * @brief Performs the forward pass of the convolution.
     */
    Tensor forward(const Tensor& input_tensor) {
        cached_input = input_tensor; // Cache for backpropagation
        const int height = input_tensor[0].rows();
        const int width = input_tensor[0].cols();
        const int padding = kernel_size / 2;

        // Pad the input tensor to handle borders
        Tensor padded_input(input_channels);
        for (int i = 0; i < input_channels; ++i) {
            padded_input[i] = pad_with_zeros(input_tensor[i], padding);
        }

        Tensor output_tensor(output_channels, Eigen::MatrixXf::Zero(height, width));
        
        for (int out_idx = 0; out_idx < output_channels; ++out_idx) {
            Eigen::MatrixXf accumulator = Eigen::MatrixXf::Constant(height, width, biases(out_idx));
            
            // Convolve with each input channel and accumulate the results
            for (int in_idx = 0; in_idx < input_channels; ++in_idx) {
                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        // Element-wise product of the kernel and the input patch, then sum
                        accumulator(row, col) += (padded_input[in_idx].block(row, col, kernel_size, kernel_size)
                                                  .cwiseProduct(weights[out_idx][in_idx])).sum();
                    }
                }
            }
            output_tensor[out_idx] = accumulator;
        }
        return output_tensor;
    }

    /**
     * @brief Performs the backward pass (backpropagation) and updates weights.
     */
    Tensor backward(const Tensor& output_gradient) {
        const int height = cached_input[0].rows();
        const int width = cached_input[0].cols();
        const int padding = kernel_size / 2;

        std::vector<Tensor> weights_gradient(output_channels, Tensor(input_channels, Eigen::MatrixXf::Zero(kernel_size, kernel_size)));
        Eigen::VectorXf biases_gradient = Eigen::VectorXf::Zero(output_channels);
        Tensor input_gradient(input_channels, Eigen::MatrixXf::Zero(height, width));

        // Pad the cached input (from the forward pass)
        Tensor padded_input(input_channels);
        for (int i = 0; i < input_channels; ++i) {
            padded_input[i] = pad_with_zeros(cached_input[i], padding);
        }

        // --- Calculate gradients for weights and biases --- //
        for (int out_idx = 0; out_idx < output_channels; ++out_idx) {
            biases_gradient(out_idx) += output_gradient[out_idx].sum();
            for (int in_idx = 0; in_idx < input_channels; ++in_idx) {
                // The gradient for the weights is a convolution between the padded input and the output gradient.
                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        weights_gradient[out_idx][in_idx] += output_gradient[out_idx](row, col) * padded_input[in_idx].block(row, col, kernel_size, kernel_size);
                    }
                }
            }
        }

        //Calculate gradient for the input //
        Tensor padded_output_gradient(output_channels);
        for (int i = 0; i < output_channels; ++i) {
            padded_output_gradient[i] = pad_with_zeros(output_gradient[i], padding);
        }

        for (int in_idx = 0; in_idx < input_channels; ++in_idx) {
            Eigen::MatrixXf accumulator = Eigen::MatrixXf::Zero(height, width);
            for (int out_idx = 0; out_idx < output_channels; ++out_idx) {
                // The gradient for the input is a "full" convolution, which requires flipping the kernel 180 degrees.
                Eigen::MatrixXf flipped_kernel = weights[out_idx][in_idx].colwise().reverse().rowwise().reverse();
                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        accumulator(row, col) += (padded_output_gradient[out_idx].block(row, col, kernel_size, kernel_size)
                                                  .cwiseProduct(flipped_kernel)).sum();
                    }
                }
            }
            input_gradient[in_idx] = accumulator;
        }

        // --- Update parameters using gradient descent --- //
        for (int out_idx = 0; out_idx < output_channels; ++out_idx) {
            biases(out_idx) -= learning_rate * biases_gradient(out_idx);
            for (int in_idx = 0; in_idx < input_channels; ++in_idx) {
                weights[out_idx][in_idx] -= learning_rate * weights_gradient[out_idx][in_idx];
            }
        }

        return input_gradient;
    }
};