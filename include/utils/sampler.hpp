#pragma once

#include <vector>
#include <numeric>   // For std::iota
#include <random>    // For std::mt19937
#include <algorithm> // For std::shuffle

/**
 * @brief A utility to sample indices from a range without replacement.
 */
struct Sampler {
private:
    std::vector<int> indices;          
    size_t current_position = 0;       
    std::mt19937 random_generator{1337};

public:
    explicit Sampler(int num_items) {
        // Create a vector of indices [0, 1, 2, ..., N-1].
        indices.resize(num_items);
        std::iota(indices.begin(), indices.end(), 0);

        // Perform an initial shuffle.
        std::shuffle(indices.begin(), indices.end(), random_generator);
    }

    /**
     * @brief Gets the next random index from the sequence.
     */
    int next() {
        // Re-shuffle the indices and reset the position to the beginning.
        if (current_position >= indices.size()) {
            std::shuffle(indices.begin(), indices.end(), random_generator);
            current_position = 0;
        }

        return indices[current_position++];
    }
};