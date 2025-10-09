#pragma once

#include <Eigen/Dense>
#include <vector>


struct ReLU2D {
    using Tensor = std::vector<Eigen::MatrixXf>;
    Tensor cached_input;

    /**
     * @brief Applies the ReLU activation function element-wise.
     */
    Tensor forward(const Tensor& input_tensor) {
        cached_input = input_tensor;
        Tensor output_tensor = input_tensor; // Make a copy
        
        for (auto& matrix : output_tensor) {
            matrix = matrix.cwiseMax(0.0f);
        }
        return output_tensor;
    }

    /**
     * @brief Computes the gradient of the ReLU function.
     */
    Tensor backward(const Tensor& output_gradient) {
        Tensor input_gradient = output_gradient; // Make a copy
        
        for (size_t channel_idx = 0; channel_idx < input_gradient.size(); ++channel_idx) {
            auto mask = (cached_input[channel_idx].array() > 0).cast<float>();
            
            // Apply the mask element-wise to the incoming gradient.
            input_gradient[channel_idx] = input_gradient[channel_idx].cwiseProduct(mask.matrix());
        }
        return input_gradient;
    }
};

struct ReLU1D {
    Eigen::VectorXf cached_input;

    /**
     * @brief Applies the ReLU activation function element-wise to a vector.
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& input_vector) {
        cached_input = input_vector;
        return input_vector.cwiseMax(0.0f);
    }

    /**
     * @brief Computes the gradient of the ReLU function for a vector.
     */
    Eigen::VectorXf backward(const Eigen::VectorXf& output_gradient) {
        Eigen::VectorXf mask = (cached_input.array() > 0).cast<float>();
        
        // Apply the mask to the incoming gradient.
        return output_gradient.cwiseProduct(mask);
    }
};