#pragma once

#include <algorithm>
#include <cmath>
#include <random>

#include "Matrix.hpp"
#include "Util.hpp"

namespace nn
{

enum class InitType
{
    XavierUniform,
    XavierNormal
};

/* Xavier（Glorot）初始化：fan_in = 行数，fan_out = 列数 */
inline void initialize(Matrix& w, InitType type)
{
    const float fan_in = static_cast<float>(w.rows());
    const float fan_out = static_cast<float>(w.cols());
    std::mt19937& gen = rng();

    if (type == InitType::XavierUniform)
    {
        const float limit = std::sqrt(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::generate(w.begin(), w.end(), [&] { return dist(gen); });
    }
    else /* InitType::XavierNormal */
    {
        const float stddev = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, stddev);
        std::generate(w.begin(), w.end(), [&] { return dist(gen); });
    }
}

} // namespace nn
