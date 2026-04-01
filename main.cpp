#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <iomanip> // For controlling output formatting

// Data type definitions
using Vector = std::vector<float>;
using Matrix = std::vector<Vector>;

// --- Full loadCSV function ---
Matrix loadCSV(const std::string &filename)
{
    Matrix data_matrix;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("ERROR: Could not open file: " + filename);
    }
    std::string line;
    while (std::getline(file, line))
    {
        Vector current_row;
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ','))
        {
            current_row.push_back(std::stof(cell));
        }
        data_matrix.push_back(current_row);
    }
    file.close();
    return data_matrix;
}

// --- Full multiply function ---
Vector multiply(const Matrix &matrix, const Vector &vector)
{
    size_t matrix_rows = matrix.size();
    size_t matrix_cols = matrix[0].size();
    if (matrix_cols != vector.size())
    {
        throw std::runtime_error("ERROR: Matrix and vector dimensions are not compatible for multiplication!");
    }
    Vector result(matrix_rows, 0.0f);
    for (size_t i = 0; i < matrix_rows; ++i)
    {
        for (size_t j = 0; j < matrix_cols; ++j)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

int main()
{
    try
    {
        // --- 1. Load ALL Data ---
        Matrix weights1 = loadCSV("cpp_project_data/weights1.csv");
        Matrix biases1_m = loadCSV("cpp_project_data/biases1.csv");
        Matrix weights2 = loadCSV("cpp_project_data/weights2.csv");
        Matrix biases2_m = loadCSV("cpp_project_data/biases2.csv");
        Matrix image_m = loadCSV("cpp_project_data/test_image.csv");

        Vector biases1, biases2, image;
        for (const auto &row : biases1_m)
        {
            biases1.push_back(row[0]);
        }
        for (const auto &row : biases2_m)
        {
            biases2.push_back(row[0]);
        }
        for (const auto &row : image_m)
        {
            image.push_back(row[0]);
        }

        // --- 2. FORWARD PASS: Layer 1 ---
        Vector layer1_output = multiply(weights1, image);
        for (size_t i = 0; i < layer1_output.size(); ++i)
        {
            layer1_output[i] += biases1[i];
        }
        for (auto &val : layer1_output)
        {
            if (val < 0)
                val = 0;
        } // ReLU

        // --- 3. FORWARD PASS: Layer 2 (Output Layer) ---
        Vector final_output = multiply(weights2, layer1_output);
        for (size_t i = 0; i < final_output.size(); ++i)
        {
            final_output[i] += biases2[i];
        }

        // --- 4. Print the raw scores (logits) for each digit ---
        std::cout << "\n--- Final Output Scores (Logits) ---" << std::endl;
        for (size_t i = 0; i < final_output.size(); ++i)
        {
            std::cout << "Digit " << i << ": " << std::fixed << std::setprecision(4) << final_output[i] << std::endl;
        }

        // --- 5. Find the Prediction ---
        auto max_it = std::max_element(final_output.begin(), final_output.end());
        int prediction = std::distance(final_output.begin(), max_it);

        std::cout << "\n========================================" << std::endl;
        std::cout << "The network predicts the digit is: " << prediction << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "A critical error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}