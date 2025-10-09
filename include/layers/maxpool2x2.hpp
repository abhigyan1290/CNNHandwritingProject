#pragma once

#include <Eigen/Dense>
#include <vector>


struct MaxPool2x2 {
    using Tensor = std::vector<Eigen::MatrixXf>;


    Tensor max_location_mask;
    int cached_input_height = 0;
    int cached_input_width = 0;

    /**
     * @brief Performs the forward pass, downsampling the input by taking the max value in 2x2 windows.
     */
    Tensor forward(const Tensor& input_tensor) {
        cached_input_height = input_tensor[0].rows();
        cached_input_width = input_tensor[0].cols();
        max_location_mask.assign(input_tensor.size(), Eigen::MatrixXf::Zero(cached_input_height, cached_input_width));

        const int output_height = cached_input_height / 2;
        const int output_width = cached_input_width / 2;
        Tensor output_tensor(input_tensor.size(), Eigen::MatrixXf::Zero(output_height, output_width));

        for (size_t channel_idx = 0; channel_idx < input_tensor.size(); ++channel_idx) {
            for (int row = 0; row < output_height; ++row) {
                for (int col = 0; col < output_width; ++col) {
                    // Extract the 2x2 block from the input tensor.
                    auto input_block = input_tensor[channel_idx].block(row * 2, col * 2, 2, 2);

                    // Find the max value and its relative indices within the 2x2 block.
                    float max_value;
                    Eigen::Index relative_row_idx, relative_col_idx;
                    max_value = input_block.maxCoeff(&relative_row_idx, &relative_col_idx);
                    output_tensor[channel_idx](row, col) = max_value;

                    // Record the absolute position of the max value in the mask for backpropagation.
                    max_location_mask[channel_idx](row * 2 + relative_row_idx, col * 2 + relative_col_idx) = 1.0f;
                }
            }
        }
        return output_tensor;
    }

    /**
     * @brief Performs the backward pass, routing gradients to the locations of the max values.
     */
    Tensor backward(const Tensor& output_gradient) {
        const int output_height = cached_input_height / 2;
        const int output_width = cached_input_width / 2;
        Tensor input_gradient(max_location_mask.size(), Eigen::MatrixXf::Zero(cached_input_height, cached_input_width));

        for (size_t channel_idx = 0; channel_idx < max_location_mask.size(); ++channel_idx) {
            for (int row = 0; row < output_height; ++row) {
                for (int col = 0; col < output_width; ++col) {
                    // Iterate through the original 2x2 block in the input space.
                    for (int block_row = 0; block_row < 2; ++block_row) {
                        for (int block_col = 0; block_col < 2; ++block_col) {
                            int absolute_row = row * 2 + block_row;
                            int absolute_col = col * 2 + block_col;

                            // If this position was the maximum (as recorded in the mask),
                            // pass the gradient from the output back to this position.
                            if (max_location_mask[channel_idx](absolute_row, absolute_col) > 0.5f) {
                                input_gradient[channel_idx](absolute_row, absolute_col) = output_gradient[channel_idx](row, col);
                            }
                        }
                    }
                }
            }
        }
        return input_gradient;
    }
};