#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include "vectorIndex.h"

using Clock = std::chrono::high_resolution_clock;



void load_embeddings_csv(const std:: string& filename, VectorIndex vi) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open embeddings file: " + filename);
    }
    std::string line;
    int line_num = 0;

    while(std::getline(f, line)) {

        if (line.empty()) {
            continue;
        }
        
        ++line_num;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> chunk;
    
        //first line is index and not necessary
        if (!std::getline(ss, cell, ',')) {
            throw std::runtime_error("Malformed CSV at line " + std::to_string(line_num));
        }
        
        while (std::getline(ss, cell, ',')) {
            try {
                chunk.push_back(std::stof(cell));
            } 
            catch (const std::exception&) {
                throw std::runtime_error(
                    "Invalid float in CSV at line " + std::to_string(line_num) +
                    ": \"" + cell + "\""
                );
            }
        }

        if (chunk.empty()) {
            throw std::runtime_error("No embedding values found at line " + std::to_string(line_num));
        }
        vi.add_vector(chunk);
    }
}



static std::string metric_to_string(Metric m) {
    switch (m) {
        case Metric::L2:     return "L2";
        case Metric::L1:     return "L1";
        case Metric::COSINE: return "COSINE";
    }
    return "UNKNOWN";
}

static void print_results(const std::vector<SearchResult>& results) {
    for (const auto& r : results) {
        std::cout << "  id=" << r.id << ", score=" << r.score << '\n';
    }
}

static void small_correctness_test() {
    std::cout << "===== SMALL CORRECTNESS TEST =====\n";

    VectorIndex index(3);

    index.add_vector({1.0f, 0.0f, 0.0f}); // id 0
    index.add_vector({0.0f, 1.0f, 0.0f}); // id 1
    index.add_vector({1.0f, 1.0f, 0.0f}); // id 2
    index.add_vector({0.0f, 0.0f, 1.0f}); // id 3

    std::vector<float> query = {1.0f, 0.0f, 0.0f};

    for (Metric m : {Metric::L2, Metric::L1, Metric::COSINE}) {
        std::cout << "\nMetric: " << metric_to_string(m) << '\n';
        auto results = index.k_closest(query, 4, m);
        print_results(results);
    }

    std::cout << '\n';
}

static std::vector<float> random_vector(std::mt19937& rng, size_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i) {
        v[i] = dist(rng);
    }

    // Avoid zero vector for cosine normalization
    float sum = 0.0f;
    for (float x : v) {
        sum += x * x;
    }
    if (sum < 1e-12f) {
        v[0] = 1.0f;
    }

    return v;
}

static void benchmark(size_t num_vectors, size_t dimension, size_t k, size_t num_queries) {
    std::cout << "===== BENCHMARK =====\n";
    std::cout << "num_vectors = " << num_vectors
              << ", dimension = " << dimension
              << ", k = " << k
              << ", num_queries = " << num_queries << "\n\n";

    std::mt19937 rng(42);

    VectorIndex index(dimension);

    auto insert_start = Clock::now();
    for (size_t i = 0; i < num_vectors; ++i) {
        index.add_vector(random_vector(rng, dimension));
    }
    auto insert_end = Clock::now();

    const auto insert_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - insert_start).count();

    std::cout << "Insertion time: " << insert_ms << " ms\n\n";

    for (Metric m : {Metric::L2, Metric::L1, Metric::COSINE}) {
        std::vector<float> query = random_vector(rng, dimension);

        // Warm-up
        index.k_closest(query, k, m);

        auto query_start = Clock::now();
        for (size_t q = 0; q < num_queries; ++q) {
            query = random_vector(rng, dimension);
            index.k_closest(query, k, m);
        }
        auto query_end = Clock::now();

        const auto total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start).count();

        const double avg_us = static_cast<double>(total_us) / static_cast<double>(num_queries);

        std::cout << std::left << std::setw(8) << metric_to_string(m)
                  << " total = " << total_us << " us"
                  << ", avg/query = " << avg_us << " us\n";
    }

    std::cout << '\n';
}

static void benchmark_serial_vs_parallel(size_t num_vectors,
                                         size_t dimension,
                                         size_t k,
                                         size_t num_queries) {
    std::cout << "===== SERIAL VS PARALLEL BENCHMARK =====\n";
    std::cout << "num_vectors = " << num_vectors
              << ", dimension = " << dimension
              << ", k = " << k
              << ", num_queries = " << num_queries << "\n\n";

    std::mt19937 rng(42);
    VectorIndex index(dimension);

    for (size_t i = 0; i < num_vectors; ++i) {
        index.add_vector(random_vector(rng, dimension));
    }

    for (Metric m : {Metric::L2, Metric::L1, Metric::COSINE}) {
        std::vector<float> query = random_vector(rng, dimension);

        // Warm-up
        index.k_closest(query, k, m);
        index.k_closest_parallel(query, k, m);

        auto serial_start = Clock::now();
        for (size_t q = 0; q < num_queries; ++q) {
            query = random_vector(rng, dimension);
            index.k_closest(query, k, m);
        }
        auto serial_end = Clock::now();

        auto parallel_start = Clock::now();
        for (size_t q = 0; q < num_queries; ++q) {
            query = random_vector(rng, dimension);
            index.k_closest_parallel(query, k, m);
        }
        auto parallel_end = Clock::now();

        const auto serial_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                serial_end - serial_start).count();

        const auto parallel_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                parallel_end - parallel_start).count();

        const double serial_avg =
            static_cast<double>(serial_us) / static_cast<double>(num_queries);

        const double parallel_avg =
            static_cast<double>(parallel_us) / static_cast<double>(num_queries);

        const double speedup =
            parallel_us > 0 ? static_cast<double>(serial_us) / static_cast<double>(parallel_us)
                            : 0.0;

        std::cout << std::left << std::setw(8) << metric_to_string(m)
                  << " serial avg/query = " << serial_avg << " us"
                  << ", parallel avg/query = " << parallel_avg << " us"
                  << ", speedup = " << speedup << "x\n";
    }

    std::cout << '\n';
}

static void edge_case_tests() {
    std::cout << "===== EDGE CASE TESTS =====\n";

    VectorIndex index(3);
    index.add_vector({1.0f, 2.0f, 3.0f});
    index.add_vector({4.0f, 5.0f, 6.0f});

    try {
        index.add_vector({1.0f, 2.0f});
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected add_vector error: " << e.what() << '\n';
    }

    try {
        index.k_closest({1.0f, 2.0f}, 1, Metric::L2);
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected query dimension error: " << e.what() << '\n';
    }

    try {
        index.k_closest({0.0f, 0.0f, 0.0f}, 1, Metric::COSINE);
    }
    catch (const std::exception& e) {
        std::cout << "Caught expected zero-vector cosine error: " << e.what() << '\n';
    }

    auto results = index.k_closest({1.0f, 2.0f, 3.0f}, 10, Metric::L2);
    std::cout << "k > numVectors returned " << results.size() << " results\n\n";
}

int main() {
    try {
        small_correctness_test();
        edge_case_tests();

        // You can change these numbers as needed.
        benchmark_serial_vs_parallel(10000, 128, 5, 50);
        benchmark_serial_vs_parallel(50000, 128, 5, 20);
        benchmark_serial_vs_parallel(5000000, 128, 5, 20);
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}