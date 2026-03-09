#pragma once

#include <string>
#include <sstream>

#include "vectorIndex.h"

struct ChunkInfo {
    size_t id;
    std::string source;
    std::string text;
};

class ChunkStore{
    public:

        /**
         * @brief Load chunk metadata from a JSON file.
         * Parses a JSON file produced by the embedding pipeline and extracts
         * chunk metadata (id, source, and text). Each chunk is stored internally
         * and can later be retrieved by id using `get()`.
         * @param filename Path to the JSON file containing chunk metadata.
         * @throws std::runtime_error
         * @note The JSON is expected to contain sequential ids starting from 0.
         */
        void load_chunks_json(const std::string& filename);
        const ChunkInfo& get(size_t id) const;
        size_t size() const;
        


    private:
        std::vector<ChunkInfo> chunks;

};