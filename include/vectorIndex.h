#pragma once
#include <vector>

struct SearchResult {
    size_t id;
    double score;
};

enum class Metric {
    L2,
    L1,
    COSINE
};

class VectorIndex{
    public:
        VectorIndex(size_t d);

        std::vector<float> get_data() const;
        size_t get_numVectors() const;
        size_t get_dimension() const;

        void set_data(std::vector<float> d);
        void set_numVectors(size_t nv);
        void set_dimension(size_t d);

        void add_vector(std::vector<float> v);
        
        void normalize_inplace(std::vector<float>& v);
        std::vector<float> normalized(std::vector<float> v);


        double squared_l2(size_t i1, size_t i2) const;
        double squared_l2(const std::vector<float>& v1, size_t i1) const;
        double squared_l2(const std::vector<float>& v1, const std::vector<float>& v2) const;
        
        float cosine_similarity(size_t i1, size_t i2) const;
        float cosine_similarity(std::vector<float> v1, size_t i1);
        float cosine_similarity(std::vector<float> v1, std::vector<float> v2);
       
        float manhattan_distance(size_t i1, size_t i2) const;
        float manhattan_distance(const std::vector<float>& v1, size_t i1) const;
        float manhattan_distance(const std::vector<float>& v1, const std::vector<float>& v2) const;

        std::vector<SearchResult> k_closest(std::vector<float> v,size_t k, Metric m);
        std::vector<SearchResult> k_closest_parallel(std::vector<float> v, size_t k, Metric m);
        
        
    private:
        std::vector<float> data;
        std::vector<float> data_normalized;
        size_t numVectors;
        size_t dimension;
};


