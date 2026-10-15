#pragma once

#include <cassert>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <iostream>
#include <string_view>
#include <vector>

#include "Activation.hpp"
#include "Architecture.hpp"
#include "Matrix.hpp"

namespace nn
{

class NN
{
public:
    using size_type = std::size_t;

    explicit NN(const Architecture& arch)
    {
        const std::vector<size_type> sizes = arch.sizes();
        assert(sizes.size() >= 2);
        count_ = sizes.size() - 1;

        ws_.reserve(count_);
        bs_.reserve(count_);
        as_.reserve(count_ + 1);

        as_.emplace_back(1, sizes[0]);
        for (size_type i = 1; i < sizes.size(); ++i)
        {
            ws_.emplace_back(as_[i - 1].cols(), sizes[i]);
            bs_.emplace_back(1, sizes[i]);
            as_.emplace_back(1, sizes[i]);
        }
    }

    NN(std::initializer_list<size_type> sizes)
        : NN(Architecture::from_sizes(sizes)) {}

    size_type count() const noexcept { return count_; }

    Matrix& input() { return as_[0]; }
    Matrix& output() { return as_[count_]; }
    const Matrix& input() const { return as_[0]; }
    const Matrix& output() const { return as_[count_]; }

    void randomize(float low, float high)
    {
        for (size_type i = 0; i < count_; ++i)
        {
            nn::randomize(ws_[i], low, high);
            nn::randomize(bs_[i], low, high);
        }
    }

    void print(std::string_view name) const
    {
        std::cout << std::format("{} = [\n", name);
        for (size_type i = 0; i < count_; ++i)
        {
            nn::print(ws_[i], std::format("ws{}", i), 4);
            nn::print(bs_[i], std::format("bs{}", i), 4);
        }
        std::cout << "]\n";
    }

    void forward()
    {
        for (size_type i = 0; i < count_; ++i)
        {
            dot(as_[i + 1], as_[i], ws_[i]);
            sum(as_[i + 1], bs_[i]);
            sigmoid(as_[i + 1]);
        }
    }

    [[nodiscard]] float cost(const Matrix& ti, const Matrix& to)
    {
        assert(ti.rows() == to.rows());
        assert(to.cols() == output().cols());
        size_type n = ti.rows();

        float c = 0.f;
        for (size_type i = 0; i < n; ++i)
        {
            const RowView x = ti.row(i);
            const RowView y = to.row(i);
            copy(input(), x);
            forward();

            for (size_type j = 0; j < to.cols(); ++j)
            {
                float d = output()(0, j) - y[j];
                c += d * d;
            }
        }

        return c / static_cast<float>(n);
    }

    void backprop(const Matrix& ti, const Matrix& to, NN& g)
    {
        size_type n = ti.rows();
        size_type L = count_;

        for (size_type l = 0; l < L; ++l)
        {
            fill(g.ws_[l], 0.0f);
            fill(g.bs_[l], 0.0f);
        }

        std::vector<Matrix> dt;
        dt.reserve(L + 1);
        dt.emplace_back();
        for (size_type l = 1; l <= L; ++l)
        {
            dt.emplace_back(1, as_[l].cols());
        }

        for (size_type sample = 0; sample < n; ++sample)
        {
            const RowView x = ti.row(sample);
            copy(input(), x);
            forward();

            const RowView y = to.row(sample);
            for (size_type j = 0; j < as_[L].cols(); ++j)
            {
                float out = as_[L](0, j);
                float target = y[j];
                float grad = 2.0f * (out - target) * out * (1.0f - out);
                dt[L](0, j) = grad;
            }

            for (size_type l = L; l >= 1; --l)
            {
                Matrix& a_prev = as_[l - 1];
                Matrix& d_cur = dt[l];

                for (size_type i = 0; i < a_prev.cols(); ++i)
                {
                    for (size_type j = 0; j < d_cur.cols(); ++j)
                    {
                        g.ws_[l - 1](i, j) += a_prev(0, i) * d_cur(0, j);
                    }
                }
                for (size_type j = 0; j < d_cur.cols(); ++j)
                {
                    g.bs_[l - 1](0, j) += d_cur(0, j);
                }

                if (l > 1)
                {
                    for (size_type i = 0; i < a_prev.cols(); ++i)
                    {
                        float s = 0.0f;
                        for (size_type j = 0; j < d_cur.cols(); ++j)
                        {
                            s += d_cur(0, j) * ws_[l - 1](i, j);
                        }
                        float a = a_prev(0, i);
                        dt[l - 1](0, i) = s * a * (1.0f - a);
                    }
                }
            }
        }

        for (size_type l = 0; l < L; ++l)
        {
            for (size_type i = 0; i < ws_[l].rows(); ++i)
            {
                for (size_type j = 0; j < ws_[l].cols(); ++j)
                {
                    g.ws_[l](i, j) /= static_cast<float>(n);
                }
            }
            for (size_type j = 0; j < bs_[l].cols(); ++j)
            {
                g.bs_[l](0, j) /= static_cast<float>(n);
            }
        }
    }

    void finite_diff(NN& g, float eps, const Matrix& ti, const Matrix& to)
    {
        float c = cost(ti, to);

        for (size_type i = 0; i < count_; ++i)
        {
            for (size_type j = 0; j < ws_[i].rows(); ++j)
            {
                for (size_type k = 0; k < ws_[i].cols(); ++k)
                {
                    float saved = ws_[i](j, k);
                    ws_[i](j, k) += eps;
                    g.ws_[i](j, k) = (cost(ti, to) - c) / eps;
                    ws_[i](j, k) = saved;
                }
            }

            for (size_type j = 0; j < bs_[i].rows(); ++j)
            {
                for (size_type k = 0; k < bs_[i].cols(); ++k)
                {
                    float saved = bs_[i](j, k);
                    bs_[i](j, k) += eps;
                    g.bs_[i](j, k) = (cost(ti, to) - c) / eps;
                    bs_[i](j, k) = saved;
                }
            }
        }
    }

    void learn(const NN& g, float rate)
    {
        for (size_type i = 0; i < count_; ++i)
        {
            for (size_type j = 0; j < ws_[i].rows(); ++j)
            {
                for (size_type k = 0; k < ws_[i].cols(); ++k)
                {
                    ws_[i](j, k) -= rate * g.ws_[i](j, k);
                }
            }
            for (size_type j = 0; j < bs_[i].rows(); ++j)
            {
                for (size_type k = 0; k < bs_[i].cols(); ++k)
                {
                    bs_[i](j, k) -= rate * g.bs_[i](j, k);
                }
            }
        }
    }

private:
    size_type count_{0};
    std::vector<Matrix> ws_;
    std::vector<Matrix> bs_;
    std::vector<Matrix> as_;
};

} // namespace nn
