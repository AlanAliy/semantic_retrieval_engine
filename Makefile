CXX := g++-15
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -pedantic -fopenmp -Iinclude
LDFLAGS := -fopenmp

TARGET := build/semantic_search

SRCS := src/main.cpp \
        src/vectorIndex.cpp \
        src/chunkStore.cpp \
        src/embeddingLoader.cpp

OBJS := $(SRCS:src/%.cpp=build/%.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p build
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

build/%.o: src/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) data/processed/embeddings.csv data/processed/chunks.json data/processed/query.csv

clean:
	rm -rf build