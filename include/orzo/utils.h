#ifndef UTILS_H
#define UTILS_H

#include <functional>
#include <random>
#include <chrono>
#include <iostream>
#include <bit>

double benchmark(std::function<void(void)> cb) {
    auto start = std::chrono::high_resolution_clock::now();
    cb();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
}

template<typename T>
T random_real(T a, T b) {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    std::uniform_real_distribution<T> dist(a, b);
    return dist(rng);
}

template<typename T = std::mt19937::result_type>
T random_integer(T a, T b, size_t seed = 42) {
    static std::random_device dev;
    static std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(a, b);
    return dist(rng);
}

template<typename T>
void print_bits(T x, size_t max = sizeof(T) * 8) {
    for (int i = max - 1; i >= 0; i--) {
        std::cout << ((x & (T(1) << i)) ? "1" : "0");
    }
}

#endif /* UTILS_H */
