#pragma once

#include <cstdint>
#include <random>

namespace nn
{

inline std::mt19937& rng()
{
    static thread_local std::mt19937 engine{std::random_device{}()};
    return engine;
}

inline void seed_rng(std::uint32_t seed)
{
    rng().seed(seed);
}

} // namespace nn
