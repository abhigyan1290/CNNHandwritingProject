#include <iostream>
#include <string>
#include <vector>
#include <algorithm> // For std::min

#include <Eigen/Dense>

#include "utils/mnist_io.hpp"
#include "utils/sampler.hpp"
#include "model.hpp"

#ifdef OPENCV_OK
#include <opencv2/opencv.hpp>
#endif

/**
 * @brief Evaluates the accuracy of the CNN model on a given dataset.
 */
static float evaluate_accuracy(CNN& model, const utils::MNIST& dataset, int sample_limit = -1) {
    int num_samples_to_test;
    if (sample_limit > 0) {
        num_samples_to_test = std::min(sample_limit, static_cast<int>(dataset.images.size()));
    } else {
        num_samples_to_test = static_cast<int>(dataset.images.size());
    }

    if (num_samples_to_test == 0) {
        return 0.0f;
    }

    int correct_predictions = 0;
    for (int i = 0; i < num_samples_to_test; ++i) {
        if (model.predict(dataset.images[i]) == dataset.labels[i]) {
            correct_predictions++;
        }
    }

    return 100.0f * static_cast<float>(correct_predictions) / num_samples_to_test;
}

/**
 * @brief Main entry point for training and evaluating the CNN on the MNIST dataset.
 */
int main(int argc, char** argv) {
    // --- Argument Parsing ---
    if (argc < 5) {
        std::cerr << "Usage: ./mnist_cnn <train_images_path> <train_labels_path> <test_images_path> <test_labels_path>\n";
        return 1;
    }

    // --- Data Loading ---
    std::cout << "Loading MNIST dataset...\n";
    auto training_data = utils::load_mnist_dataset(argv[1], argv[2]);
    auto testing_data = utils::load_mnist_dataset(argv[3], argv[4]);
    std::cout << "Dataset loaded successfully.\n"
              << "  - Training samples: " << training_data.images.size() << "\n"
              << "  - Testing samples:  " << testing_data.images.size() << "\n\n";

    // --- Hyperparameters and Model Initialization ---
    const int NUM_EPOCHS = 2;
    const float LEARNING_RATE = 0.01f;
    const int STEPS_PER_EPOCH = training_data.images.size(); // One full pass over the data
    
    CNN model(LEARNING_RATE);

    // --- Optional: Visual Sanity Check with OpenCV ---
#ifdef OPENCV_OK
    // Display the first training sample to verify data loading.
    cv::Mat image_float(28, 28, CV_32F, const_cast<float*>(training_data.images[0].data()));
    cv::Mat display_image;
    image_float.convertTo(display_image, CV_8U, 255.0); // Convert to 8-bit for display
    cv::resize(display_image, display_image, {140, 140}, 0, 0, cv::INTER_NEAREST); // Make it bigger
    cv::imshow("First Training Sample", display_image);
    cv::waitKey(500); // Display for 0.5 seconds
#endif

    // --- Training Loop ---
    std::cout << "Starting training...\n";
    Sampler sampler(static_cast<int>(training_data.images.size()));

    for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        double loss_accumulator = 0.0;
        int step_count_for_avg = 0;

        for (int step = 0; step < STEPS_PER_EPOCH; ++step) {
            // Get a random sample from the training data.
            int sample_index = sampler.next();
            
            // Perform one training step (forward and backward pass).
            float loss = model.train_step(training_data.images[sample_index], training_data.labels[sample_index]);
            
            loss_accumulator += loss;
            step_count_for_avg++;

            // Print average loss every 1000 steps as a progress update.
            if ((step + 1) % 1000 == 0) {
                std::cout << "[Epoch " << epoch << "] Step " << (step + 1)
                          << "   Avg Loss: " << (loss_accumulator / step_count_for_avg) << "\n";
                loss_accumulator = 0.0;
                step_count_for_avg = 0;
            }
        }
        
        // Evaluate accuracy on a subset of the test set after each epoch.
        float accuracy = evaluate_accuracy(model, testing_data, 1000);
        std::cout << "--- End of Epoch " << epoch << " ---\n"
                  << "Test Accuracy (on 1000 samples): " << accuracy << "%\n\n";
    }

    // --- Final Evaluation ---
    std::cout << "Training complete. Running final evaluation on the full test set...\n";
    float final_accuracy = evaluate_accuracy(model, testing_data);
    std::cout << "========================================\n"
              << "Final Test Accuracy: " << final_accuracy << "%\n"
              << "========================================\n";

    return 0;
}