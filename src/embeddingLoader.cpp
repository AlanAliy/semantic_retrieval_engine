
#include "embeddingLoader.h"
#include <string>
#include <sstream>
#include <fstream>


void load_embeddings_csv(const std:: string& filename, VectorIndex& vi) {
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
    
        //first line is id and is ignored
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
};

std::vector<float> load_query_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open query file: " + filename);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Query file is empty");
    }

    std::stringstream ss(line);
    std::string cell;
    std::vector<float> query;

    while (std::getline(ss, cell, ',')) {
        query.push_back(std::stof(cell));
    }

    return query;
}