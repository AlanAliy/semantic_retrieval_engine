#include "vectorIndex.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>


VectorIndex::VectorIndex(size_t d)
    : numVectors(0), dimension(d)
{
    if (d == 0) {
        throw std::invalid_argument("[CTOR] ERROR: Dimension must be > 0");
    }
}

std::vector<float> VectorIndex::get_data() const { return data; }
size_t VectorIndex::get_numVectors() const { return numVectors; }
size_t VectorIndex::get_dimension() const { return dimension; }

void VectorIndex::set_data(std::vector<float> d) { data = d; }
void VectorIndex::set_numVectors(size_t nv) { numVectors = nv; }
void VectorIndex::set_dimension(size_t d) { dimension = d; }

void VectorIndex::normalize_inplace(std::vector<float>& v) {
    if (v.size() != dimension) {
        throw std::invalid_argument("[NORMALIZE_INPLACE] ERROR: Vector dimensions do not match");
    }

    double sum = 0.0;
    for (float num : v) {
        sum += num * num;
    }

    // against rounding errors
    if (sum < 1e-12) {
        throw std::invalid_argument("[NORMALIZE_INPLACE] ERROR: Cannot normalize zero vector");
    }

    const double norm = std::sqrt(sum);

    for (float& num : v) {
        num /= static_cast<float>(norm);
    }
}

std::vector<float> VectorIndex::normalized(std::vector<float> v) {
    normalize_inplace(v);
    return v;
}

void VectorIndex::add_vector(std::vector<float> v) {
    if (v.size() != dimension) {
        throw std::invalid_argument("[ADD_VECTOR] ERROR: Vector dimensions do not match");
    }

    std::vector<float> v_normalized = v;
    try {
        normalize_inplace(v_normalized);
    }
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("[ADD_VECTOR] ") + e.what());
    }

    data.insert(data.end(), v.begin(), v.end());
    data_normalized.insert(data_normalized.end(), v_normalized.begin(), v_normalized.end());
    numVectors++;
}

double VectorIndex::squared_l2(size_t i1, size_t i2) const {
    if (i1 >= numVectors || i2 >= numVectors) {
        throw std::invalid_argument("[SQUARED_L2] ERROR: Vector ID out of bounds");
    }

    const size_t base1 = i1 * dimension;
    const size_t base2 = i2 * dimension;

    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        const double dif = data[base1 + i] - data[base2 + i];
        sum += dif * dif;
    }
    return sum;
}

double VectorIndex::squared_l2(const std::vector<float>& v1, size_t i1) const {
    if (v1.size() != dimension) {
        throw std::invalid_argument("[SQUARED_L2] ERROR: Vector dimensions do not match");
    }
    if (i1 >= numVectors) {
        throw std::invalid_argument("[SQUARED_L2] ERROR: Vector ID out of bounds");
    }

    const size_t base = i1 * dimension;

    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        const double dif = data[base + i] - v1[i];
        sum += dif * dif;
    }
    return sum;
}

double VectorIndex::squared_l2(const std::vector<float>& v1, const std::vector<float>& v2) const {
    if (v1.size() != dimension || v2.size() != dimension) {
        throw std::invalid_argument("[SQUARED_L2] ERROR: Vector dimensions do not match");
    }

    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        const double dif = v2[i] - v1[i];
        sum += dif * dif;
    }
    return sum;
}

float VectorIndex::cosine_similarity(std::vector<float> v1, std::vector<float> v2) {
    if (v1.size() != dimension || v2.size() != dimension) {
        throw std::invalid_argument("[cosine_similarity] ERROR: Vector dimensions do not match");
    }

    try {
        normalize_inplace(v1);
        normalize_inplace(v2);
    }
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("[cosine_similarity] ") + e.what());
    }

    float dist = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        dist += v1[i] * v2[i];
    }
    return dist;
}

float VectorIndex::cosine_similarity(std::vector<float> v1, size_t i1) {
    if (v1.size() != dimension) {
        throw std::invalid_argument("[cosine_similarity] ERROR: Vector dimensions do not match");
    }
    if (i1 >= numVectors) {
        throw std::invalid_argument("[cosine_similarity] ERROR: Vector ID out of bounds");
    }

    try {
        normalize_inplace(v1);
    }
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("[cosine_similarity] ") + e.what());
    }

    const size_t base = i1 * dimension;

    float dist = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        dist += v1[i] * data_normalized[base + i];
    }
    return dist;
}

float VectorIndex::cosine_similarity(size_t i1, size_t i2) const {
    if (i1 >= numVectors || i2 >= numVectors) {
        throw std::invalid_argument("[cosine_similarity] ERROR: Vector ID out of bounds");
    }

    const size_t base1 = i1 * dimension;
    const size_t base2 = i2 * dimension;

    float dist = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        dist += data_normalized[base1 + i] * data_normalized[base2 + i];
    }
    return dist;
}

float VectorIndex::manhattan_distance(const std::vector<float>& v1, const std::vector<float>& v2) const {
    if (v1.size() != dimension || v2.size() != dimension) {
        throw std::invalid_argument("[MANHATTAN_DISTANCE] ERROR: Vector dimensions do not match");
    }

    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        sum += std::abs(v2[i] - v1[i]);
    }
    return static_cast<float>(sum);
}

float VectorIndex::manhattan_distance(const std::vector<float>& v1, size_t i1) const {
    if (v1.size() != dimension) {
        throw std::invalid_argument("[MANHATTAN_DISTANCE] ERROR: Vector dimensions do not match");
    }
    if (i1 >= numVectors) {
        throw std::invalid_argument("[MANHATTAN_DISTANCE] ERROR: Vector ID out of bounds");
    }

    const size_t base = i1 * dimension;

    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        sum += std::abs(data[base + i] - v1[i]);
    }
    return static_cast<float>(sum);
}

float VectorIndex::manhattan_distance(size_t i1, size_t i2) const {
    if (i1 >= numVectors || i2 >= numVectors) {
        throw std::invalid_argument("[MANHATTAN_DISTANCE] ERROR: Vector ID out of bounds");
    }

    const size_t base1 = i1 * dimension;
    const size_t base2 = i2 * dimension;

    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        sum += std::abs(data[base1 + i] - data[base2 + i]);
    }
    return static_cast<float>(sum);
}

std::vector<SearchResult> VectorIndex::k_closest(std::vector<float> v, size_t k, Metric m) {
    if (v.size() != dimension) {
        throw std::invalid_argument("[K_CLOSEST] ERROR: Vector dimensions do not match");
    }

    if (k > numVectors) {
        k = numVectors;
    }

    std::vector<SearchResult> results;
    results.reserve(numVectors);

    if (m == Metric::COSINE) {
        try {
            normalize_inplace(v);
        }
        catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::string("[K_CLOSEST] ") + e.what());
        }
    }


    for (size_t id = 0; id < numVectors; ++id) {
        double score = 0.0;

        switch (m) {
            case Metric::L2:
                score = squared_l2(v, id);
                break;
            case Metric::L1:
                score = manhattan_distance(v, id);
                break;
            case Metric::COSINE:
                score = cosine_similarity(v, id);
                break;
        }

        results.push_back({id, score});
    }

    if (m == Metric::COSINE) {
        std::sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.score > b.score; // bigger cosine similarity is better
            });
    } else {
        std::sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.score < b.score; // smaller distance is better
            });
    }

    results.resize(k);
    return results;
}


std::vector<SearchResult> VectorIndex::k_closest_parallel(std::vector<float> v, size_t k, Metric m){
 if (v.size() != dimension) {
        throw std::invalid_argument("[K_CLOSEST] ERROR: Vector dimensions do not match");
    }

    if (k > numVectors) {
        k = numVectors;
    }

    std::vector<SearchResult> results(numVectors);
    results.reserve(numVectors);

    if (m == Metric::COSINE) {
        try {
            normalize_inplace(v);
        }
        catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::string("[K_CLOSEST] ") + e.what());
        }
    }

    #pragma omp parallel for
    for (long long id = 0; id < static_cast<long long>(numVectors); ++id) {
        double score = 0.0;

        switch (m) {
            case Metric::L2:
                score = squared_l2(v, static_cast<size_t>(id));
                break;
            case Metric::L1:
                score = manhattan_distance(v, static_cast<size_t>(id));
                break;
            case Metric::COSINE:
                score = cosine_similarity(v, static_cast<size_t>(id));
                break;
        }

        results[id] = {static_cast<size_t>(id), score};
    }

    if (m == Metric::COSINE) {
        std::sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.score > b.score; // bigger cosine similarity is better
            });
    } else {
        std::sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.score < b.score; // smaller distance is better
            });
    }

    results.resize(k);
    return results;
}