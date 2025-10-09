#pragma once

#include <Eigen/Dense>
#include <vector>

struct Flatten {
    using Tensor = std::vector<Eigen::MatrixXf>;

    int cached_channels = 0;
    int cached_height = 0;
    int cached_width = 0;

    /**
     * @brief Performs the forward pass, reshaping the input tensor into a vector.
     */
    Eigen::VectorXf forward(const Tensor& input_tensor) {
        // Cache the input dimensions for the backward pass.
        cached_channels = input_tensor.size();
        cached_height = input_tensor[0].rows();
        cached_width = input_tensor[0].cols();

        const int total_size = cached_channels * cached_height * cached_width;
        Eigen::VectorXf output_vector(total_size);
        
        int offset = 0;
        for (int c = 0; c < cached_channels; ++c) {
            // Use Eigen::Map to treat the matrix data as a flat vector without copying.
            Eigen::Map<const Eigen::VectorXf> channel_as_vector(
                input_tensor[c].data(),
                cached_height * cached_width
            );
            
            // Copy the flattened channel into the correct segment of the output vector.
            output_vector.segment(offset, cached_height * cached_width) = channel_as_vector;
            offset += cached_height * cached_width;
        }
        
        return output_vector;
    }

    /**
     * @brief Performs the backward pass, reshaping the gradient vector back into a tensor.
     */
    Tensor backward(const Eigen::VectorXf& output_gradient) {
        Tensor input_gradient(cached_channels, Eigen::MatrixXf(cached_height, cached_width));
        
        int offset = 0;
        for (int c = 0; c < cached_channels; ++c) {
            // Use Eigen::Map to interpret a segment of the flat gradient as a matrix.
            Eigen::Map<const Eigen::MatrixXf> gradient_as_matrix(
                output_gradient.data() + offset,
                cached_height,
                cached_width
            );
            
            // Copy the reshaped gradient into the corresponding channel of the input gradient tensor.
            input_gradient[c] = gradient_as_matrix;
            offset += cached_height * cached_width;
        }
        
        return input_gradient;
    }
};