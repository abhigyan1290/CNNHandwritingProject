#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <algorithm> // For std::max

/**
 * @brief A combined Softmax activation and Cross-Entropy loss layer.
 */
struct SoftmaxCE {
    Eigen::VectorXf cached_probabilities;
    int cached_target_label = -1;

    /**
     * @brief Computes the softmax probabilities and then the cross-entropy loss.
     */
    float forward(const Eigen::VectorXf& logits, int target_label) {
        cached_target_label = target_label;
        const float max_logit = logits.maxCoeff();
        Eigen::VectorXf exp_logits = (logits.array() - max_logit).exp();
        
        // Normalize to get probabilities.
        cached_probabilities = exp_logits / exp_logits.sum();

        // Loss is -log(p_y), where p_y is the predicted probability for the true class.
        // We clip the probability to prevent log(0), which would be -inf.
        float probability_of_target = std::max(1e-9f, cached_probabilities(target_label));
        
        return -std::log(probability_of_target);
    }

    /**
     * @brief Computes the gradient of the loss with respect to the input logits.
     */
    Eigen::VectorXf backward() {
        // gradient = probabilities - y_one_hot
        Eigen::VectorXf input_gradient = cached_probabilities;
        input_gradient(cached_target_label) -= 1.0f;
        
        return input_gradient;
    }
};