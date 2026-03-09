
#pragma once

#include "vectorIndex.h"
#include <vector>
#include <string>

/**
 * @brief Load embedding vectors from a CSV file into a VectorIndex.
 *
 * Each line of the CSV is expected to have the following format:
 *
 *     id,val1,val2,val3,...,valN
 *
 * The first column (id) is ignored. All remaining columns are parsed
 * as floating point values and inserted as a vector into the provided
 * VectorIndex via `add_vector`.
 *
 * @param filename Path to the embeddings CSV file.
 * @param vi Reference to the VectorIndex where embeddings will be stored.
 *
 * @throws std::runtime_error
 *
 * @note The function assumes that the CSV rows match the expected
 * embedding dimension used by the VectorIndex.
 */

void load_embeddings_csv(const std:: string& filename, VectorIndex& vi);
std::vector<float> load_query_csv(const std::string& filename);