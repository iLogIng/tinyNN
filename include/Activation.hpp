#pragma once

#include <algorithm>
#include <cmath>

#include "Matrix.hpp"

namespace nn
{

/* 激活函数策略：apply 为逐元素前向，derivative 以“激活后的值 a”计算导数 */

struct Sigmoid
{
    static void apply(Matrix& m)
    {
        std::transform(m.begin(), m.end(), m.begin(),
            [](float x) { return 1.f / (1.f + std::exp(-x)); });
    }

    static void derivative(const Matrix& a, Matrix& out)
    {
        std::transform(a.begin(), a.end(), out.begin(),
            [](float x) { return x * (1.f - x); });
    }
};

struct Tanh
{
    static void apply(Matrix& m)
    {
        std::transform(m.begin(), m.end(), m.begin(),
            [](float x) { return std::tanh(x); });
    }

    static void derivative(const Matrix& a, Matrix& out)
    {
        std::transform(a.begin(), a.end(), out.begin(),
            [](float x) { return 1.f - x * x; });
    }
};

struct Relu
{
    static void apply(Matrix& m)
    {
        std::transform(m.begin(), m.end(), m.begin(),
            [](float x) { return x > 0.f ? x : 0.f; });
    }

    static void derivative(const Matrix& a, Matrix& out)
    {
        std::transform(a.begin(), a.end(), out.begin(),
            [](float x) { return x > 0.f ? 1.f : 0.f; });
    }
};

enum class ActivationType
{
    Sigmoid,
    Tanh,
    Relu
};

inline void activation_forward(ActivationType type, Matrix& m)
{
    switch (type)
    {
    case ActivationType::Sigmoid: Sigmoid::apply(m); break;
    case ActivationType::Tanh:    Tanh::apply(m); break;
    case ActivationType::Relu:    Relu::apply(m); break;
    }
}

inline void activation_derivative(ActivationType type, const Matrix& a, Matrix& out)
{
    switch (type)
    {
    case ActivationType::Sigmoid: Sigmoid::derivative(a, out); break;
    case ActivationType::Tanh:    Tanh::derivative(a, out); break;
    case ActivationType::Relu:    Relu::derivative(a, out); break;
    }
}

} // namespace nn
