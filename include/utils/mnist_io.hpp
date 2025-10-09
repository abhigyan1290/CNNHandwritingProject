#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#if defined(_MSC_VER)
#include <stdlib.h> // For _byteswap_ulong
#endif

/**
 * @brief Contains utility functions for loading the MNIST dataset.
 */
namespace utils {

/**
 * @brief Reads a 32-bit unsigned integer in big-endian format from a file stream.
 * @param file_stream The input file stream to read from.
 * @return The integer value, converted to the host system's endianness.
 */
inline uint32_t read_big_endian_uint32(std::ifstream& file_stream) {
    uint32_t value;
    file_stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    
    // The MNIST dataset is stored in big-endian format, so we need to swap
    // byte order on little-endian systems (like x86).
#if defined(_MSC_VER)
    return _byteswap_ulong(value);
#else
    return __builtin_bswap32(value);
#endif
}

/**
 * @brief A struct to hold the entire MNIST dataset.
 */
struct MNIST {
    // A vector of 28x28 matrices, with pixel values normalized to [0.0, 1.0].
    std::vector<Eigen::MatrixXf> images;
    // A vector of corresponding labels, with values from 0 to 9.
    std::vector<uint8_t> labels;
};


inline MNIST load_mnist_dataset(const std::string& image_path, const std::string& label_path) {
    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file) {
        throw std::runtime_error("Failed to open image file: " + image_path);
    }

    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file) {
        throw std::runtime_error("Failed to open label file: " + label_path);
    }

    // --- Read and validate image file header --- //
    // The IDX file format starts with a "magic number" to identify the content.
    uint32_t image_magic_number = read_big_endian_uint32(image_file);
    if (image_magic_number != 2051) { // 2051 identifies an image set
        throw std::runtime_error("Invalid magic number in image file.");
    }
    uint32_t num_images = read_big_endian_uint32(image_file);
    uint32_t num_rows   = read_big_endian_uint32(image_file);
    uint32_t num_cols   = read_big_endian_uint32(image_file);
    if (num_rows != 28 || num_cols != 28) {
        throw std::runtime_error("Image dimensions are not 28x28.");
    }

    // --- Read and validate label file header --- //
    uint32_t label_magic_number = read_big_endian_uint32(label_file);
    if (label_magic_number != 2049) { // 2049 identifies a label set
        throw std::runtime_error("Invalid magic number in label file.");
    }
    uint32_t num_labels = read_big_endian_uint32(label_file);

    // Ensure the number of images and labels match.
    if (num_labels != num_images) {
        throw std::runtime_error("The number of images and labels do not match.");
    }

    // --- Read image and label data --- //
    MNIST dataset;
    dataset.images.reserve(num_images);
    dataset.labels.resize(num_labels);
    
    // Create a buffer to hold the raw pixel data for one image.
    std::vector<uint8_t> pixel_buffer(num_rows * num_cols);

    for (uint32_t i = 0; i < num_images; ++i) {
        // Read the raw pixel data.
        image_file.read(reinterpret_cast<char*>(pixel_buffer.data()), pixel_buffer.size());
        
        // Use Eigen::Map to efficiently convert the raw buffer to a matrix and normalize it.
        // This avoids a slow, manual, nested for-loop.
        Eigen::Map<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>> temp_map(pixel_buffer.data());
        dataset.images.emplace_back(temp_map.cast<float>() / 255.0f);
        
        // Read the corresponding label.
        label_file.read(reinterpret_cast<char*>(&dataset.labels[i]), sizeof(uint8_t));
    }

    return dataset;
}

} // namespace utils