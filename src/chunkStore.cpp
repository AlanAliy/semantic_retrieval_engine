#include "chunkStore.h"
#include <string>
#include <fstream>



void ChunkStore::load_chunks_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open chunks json: " + filename);
    }

    chunks.clear();

    std::string line;
    ChunkInfo current{};
    bool has_id = false;
    bool has_source = false;
    bool has_text = false;

    while (std::getline(file, line)) {
        if (line.find("\"id\"") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon == std::string::npos) {
                throw std::runtime_error("Malformed id line in " + filename);
            }

            std::string value = line.substr(colon + 1);

            while (!value.empty() &&
                   (value.back() == ',' || value.back() == ' ' || value.back() == '\n' || value.back() == '\r')) {
                value.pop_back();
            }

            current.id = std::stoul(value);
            has_id = true;
        }
        else if (line.find("\"source\"") != std::string::npos) {
            size_t first_quote = line.find('"', line.find(':') + 1);
            size_t second_quote = line.find('"', first_quote + 1);

            if (first_quote == std::string::npos || second_quote == std::string::npos) {
                throw std::runtime_error("Malformed source line in " + filename);
            }

            current.source = line.substr(first_quote + 1, second_quote - first_quote - 1);
            has_source = true;
        }
        else if (line.find("\"text\"") != std::string::npos) {
            size_t first_quote = line.find('"', line.find(':') + 1);
            size_t second_quote = line.rfind('"');

            if (first_quote == std::string::npos || second_quote == std::string::npos || second_quote <= first_quote) {
                throw std::runtime_error("Malformed text line in " + filename);
            }

            current.text = line.substr(first_quote + 1, second_quote - first_quote - 1);
            has_text = true;
        }

        if (has_id && has_source && has_text) {
            if (current.id != chunks.size()) {
                throw std::runtime_error("Chunk ids are not in sequential order");
            }

            chunks.push_back(current);

            current = ChunkInfo{};
            has_id = false;
            has_source = false;
            has_text = false;
        }
    }
}

size_t ChunkStore::size() const {
    return chunks.size();
}

const ChunkInfo& ChunkStore::get(size_t id) const {
    if (id >= chunks.size()) {
        throw std::out_of_range("Chunk id out of range");
    }

    return chunks[id];
}
