
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iostream>


#include "vectorIndex.h"
#include "embeddingLoader.h"
#include "chunkStore.h"

size_t EMBEDDING_DIM = 384;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <embeddings.csv> <chunks.json> <query.csv>\n";
        return 1;
    }

    VectorIndex vi(EMBEDDING_DIM);
    ChunkStore cs;

    load_embeddings_csv(argv[1], vi);
    cs.load_chunks_json(argv[2]);

    if (cs.size() != vi.get_numVectors()) {
        throw std::runtime_error("Chunk count and embedding count do not match");
    }

    std::vector<float> query = load_query_csv(argv[3]);

    auto results = vi.k_closest(query, 5, Metric::COSINE);

    for (const auto& r : results) {
        const ChunkInfo& chunk = cs.get(r.id);

        std::cout << "ID: " << r.id << "\n";
        std::cout << "Score: " << r.score << "\n";
        std::cout << "Source: " << chunk.source << "\n";
        std::cout << "Text: " << chunk.text << "\n\n";
    }

    return 0;
}



