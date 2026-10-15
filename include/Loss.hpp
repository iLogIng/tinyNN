#pragma once

#include <algorithm>
#include <cmath>

#include "Matrix.hpp"

namespace nn
{

/* 损失策略：value 为单样本代价，gradient 为 dL/dpred */
struct MeanSquaredError
{
    static float value(const Matrix& pred, RowView target)
    {
        float s = 0.0f;
        for (Matrix::size_type j = 0; j < pred.cols(); ++j)
        {
            float d = pred(0, j) - target[j];
            s += d * d;
        }
        return s;
    }

    static void gradient(const Matrix& pred, RowView target, Matrix& out)
    {
        for (Matrix::size_type j = 0; j < pred.cols(); ++j)
        {
            out(0, j) = 2.0f * (pred(0, j) - target[j]);
        }
    }
};

/* 二元交叉熵：期望输出层为 sigmoid（pred 位于开区间 (0,1)） */
struct BinaryCrossEntropy
{
    static float value(const Matrix& pred, RowView target)
    {
        const float eps = 1e-7f;
        float s = 0.0f;
        for (Matrix::size_type j = 0; j < pred.cols(); ++j)
        {
            float p = std::clamp(pred(0, j), eps, 1.0f - eps);
            float t = target[j];
            s += -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
        }
        return s;
    }

    static void gradient(const Matrix& pred, RowView target, Matrix& out)
    {
        const float eps = 1e-7f;
        for (Matrix::size_type j = 0; j < pred.cols(); ++j)
        {
            float p = std::clamp(pred(0, j), eps, 1.0f - eps);
            float t = target[j];
            out(0, j) = (p - t) / (p * (1.0f - p));
        }
    }
};

enum class LossType
{
    MeanSquaredError,
    BinaryCrossEntropy
};

inline float loss_value(LossType type, const Matrix& pred, RowView target)
{
    switch (type)
    {
    case LossType::MeanSquaredError:   return MeanSquaredError::value(pred, target);
    case LossType::BinaryCrossEntropy: return BinaryCrossEntropy::value(pred, target);
    }
    return 0.0f;
}

inline void loss_gradient(LossType type, const Matrix& pred, RowView target, Matrix& out)
{
    switch (type)
    {
    case LossType::MeanSquaredError:   MeanSquaredError::gradient(pred, target, out); break;
    case LossType::BinaryCrossEntropy: BinaryCrossEntropy::gradient(pred, target, out); break;
    }
}

} // namespace nn
