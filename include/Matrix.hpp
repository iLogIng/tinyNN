#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "Util.hpp"

namespace nn
{

using RowView = std::span<const float>;

class Matrix
{
public:
    using size_type = std::size_t;

    Matrix() = default;

    Matrix(size_type rows, size_type cols)
        : rows_(rows)
        , cols_(cols)
        , stride_(cols)
        , elem_(rows * cols, 0.0f) {}

    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type stride() const noexcept { return stride_; }

    float* data() noexcept { return elem_.data(); }
    const float* data() const noexcept { return elem_.data(); }

    float* begin() noexcept { return elem_.data(); }
    float* end() noexcept { return elem_.data() + elem_.size(); }
    const float* begin() const noexcept { return elem_.data(); }
    const float* end() const noexcept { return elem_.data() + elem_.size(); }

    float& operator()(size_type r, size_type c) { return elem_[r * stride_ + c]; }
    const float& operator()(size_type r, size_type c) const { return elem_[r * stride_ + c]; }

    float& at(size_type r, size_type c)
    {
        if (r >= rows_ || c >= cols_)
        {
            throw std::out_of_range("Matrix index out of range");
        }
        return elem_[r * stride_ + c];
    }

    const float& at(size_type r, size_type c) const
    {
        if (r >= rows_ || c >= cols_)
        {
            throw std::out_of_range("Matrix index out of range");
        }
        return elem_[r * stride_ + c];
    }

    [[nodiscard]] RowView row(size_type r) const noexcept
    {
        return RowView(elem_.data() + r * stride_, cols_);
    }

    void clear() noexcept { elem_.clear(); }

private:
    size_type rows_{0};
    size_type cols_{0};
    size_type stride_{0};
    std::vector<float> elem_;
};

inline void fill(Matrix& m, float val)
{
    std::fill(m.begin(), m.end(), val);
}

inline void randomize(Matrix& m, float low, float high)
{
    std::uniform_real_distribution<float> dist(low, high);
    std::mt19937& gen = rng();
    std::generate(m.begin(), m.end(), [&] { return dist(gen); });
}

inline void copy(Matrix& dst, const Matrix& src)
{
    assert(dst.rows() == src.rows());
    assert(dst.cols() == src.cols());
    std::copy(src.begin(), src.end(), dst.begin());
}

inline void copy(Matrix& dst, RowView src)
{
    assert(dst.cols() == src.size());
    std::copy(src.begin(), src.end(), dst.begin());
}

inline void dot(Matrix& dst, const Matrix& a, const Matrix& b)
{
    assert(a.cols() == b.rows());
    assert(dst.rows() == a.rows());
    assert(dst.cols() == b.cols());
    for (Matrix::size_type i = 0; i < dst.rows(); ++i)
    {
        for (Matrix::size_type j = 0; j < dst.cols(); ++j)
        {
            float acc = 0.0f;
            for (Matrix::size_type k = 0; k < a.cols(); ++k)
            {
                acc += a(i, k) * b(k, j);
            }
            dst(i, j) = acc;
        }
    }
}

inline void sum(Matrix& dst, const Matrix& a)
{
    assert(dst.rows() == a.rows());
    assert(dst.cols() == a.cols());
    std::transform(dst.begin(), dst.end(), a.begin(), dst.begin(), std::plus<float>{});
}

inline void multiply(Matrix& dst, const Matrix& a)
{
    assert(dst.rows() == a.rows());
    assert(dst.cols() == a.cols());
    std::transform(dst.begin(), dst.end(), a.begin(), dst.begin(),
        [](float x, float y) { return x * y; });
}

inline void print(const Matrix& m, std::string_view name, std::size_t padding = 0)
{
    const std::string pad(padding, ' ');
    std::string out;
    std::format_to(std::back_inserter(out), "{}{} = [\n", pad, name);
    for (Matrix::size_type i = 0; i < m.rows(); ++i)
    {
        out += "    ";
        for (Matrix::size_type j = 0; j < m.cols(); ++j)
        {
            std::format_to(std::back_inserter(out), "{}{:f}", pad, m(i, j));
        }
        out += '\n';
    }
    std::format_to(std::back_inserter(out), "{}]\n", pad);
    std::cout << out;
}

} // namespace nn
