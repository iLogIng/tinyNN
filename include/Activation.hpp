#pragma once

#include <algorithm>
#include <cmath>

#include "Matrix.hpp"

namespace nn
{

inline float sigmoidf(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

inline void sigmoid(Matrix& m)
{
    std::transform(m.begin(), m.end(), m.begin(),
        [](float x) { return sigmoidf(x); });
}

} // namespace nn
