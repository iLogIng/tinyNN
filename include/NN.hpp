#pragma once

#include <cassert>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "Activation.hpp"
#include "Architecture.hpp"
#include "Initialization.hpp"
#include "Loss.hpp"
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

    Matrix& weights(size_type i) { return ws_[i]; }
    const Matrix& weights(size_type i) const { return ws_[i]; }
    Matrix& biases(size_type i) { return bs_[i]; }
    const Matrix& biases(size_type i) const { return bs_[i]; }
    Matrix& activations(size_type i) { return as_[i]; }
    const Matrix& activations(size_type i) const { return as_[i]; }

    void set_hidden_activation(ActivationType type) noexcept { hidden_activation_ = type; }
    void set_output_activation(ActivationType type) noexcept { output_activation_ = type; }
    void set_loss(LossType type) noexcept { loss_ = type; }

    ActivationType hidden_activation() const noexcept { return hidden_activation_; }
    ActivationType output_activation() const noexcept { return output_activation_; }
    LossType loss() const noexcept { return loss_; }

    void set_weight_decay(float value)
    {
        if (value < 0.0f)
        {
            throw std::invalid_argument("NN: weight_decay must be >= 0");
        }
        weight_decay_ = value;
    }
    float weight_decay() const noexcept { return weight_decay_; }

    void set_dropout(float probability)
    {
        if (probability < 0.0f || probability >= 1.0f)
        {
            throw std::invalid_argument("NN: dropout must be in [0, 1)");
        }
        dropout_ = probability;
    }
    float dropout() const noexcept { return dropout_; }

    void set_training(bool on) noexcept { training_ = on; }
    bool training() const noexcept { return training_; }

    /* Xavier 初始化：权重按类型初始化，偏置置 0 */
    void initialize(InitType type)
    {
        for (size_type i = 0; i < count_; ++i)
        {
            nn::initialize(ws_[i], type);
            fill(bs_[i], 0.0f);
        }
    }

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
        const bool drop = training_ && dropout_ > 0.0f;
        if (drop)
        {
            ensure_dropout_storage();
        }

        for (size_type i = 0; i < count_; ++i)
        {
            dot(as_[i + 1], as_[i], ws_[i]);
            sum(as_[i + 1], bs_[i]);
            const ActivationType act = (i + 1 == count_) ? output_activation_ : hidden_activation_;
            activation_forward(act, as_[i + 1]);

            if (drop && i + 1 < count_)
            {
                const size_type l = i + 1;   // hidden layer
                copy(pre_dropout_act_[l], as_[l]);

                std::bernoulli_distribution keep(1.0 - static_cast<double>(dropout_));
                std::mt19937& gen = rng();
                const float scale = 1.0f / (1.0f - dropout_);
                for (size_type j = 0; j < as_[l].cols(); ++j)
                {
                    dropout_mask_[l](0, j) = keep(gen) ? scale : 0.0f;
                }
                multiply(as_[l], dropout_mask_[l]);
            }
        }
    }

    [[nodiscard]] float cost(const Matrix& ti, const Matrix& to)
    {
        check_shapes(ti, to);
        size_type n = ti.rows();

        float c = 0.f;
        for (size_type i = 0; i < n; ++i)
        {
            const RowView x = ti.row(i);
            const RowView y = to.row(i);
            copy(input(), x);
            forward();
            c += loss_value(loss_, output(), y);
        }

        float reg = 0.0f;
        if (weight_decay_ > 0.0f)
        {
            for (size_type l = 0; l < count_; ++l)
            {
                for (size_type i = 0; i < ws_[l].rows(); ++i)
                {
                    for (size_type j = 0; j < ws_[l].cols(); ++j)
                    {
                        reg += ws_[l](i, j) * ws_[l](i, j);
                    }
                }
            }
        }

        return c / static_cast<float>(n) + 0.5f * weight_decay_ * reg;
    }

    void backprop(const Matrix& ti, const Matrix& to, NN& g)
    {
        check_shapes(ti, to);
        if (dropout_ > 0.0f && !training_)
        {
            throw std::invalid_argument("NN: backprop with dropout requires training mode");
        }
        size_type n = ti.rows();
        size_type L = count_;

        for (size_type l = 0; l < L; ++l)
        {
            fill(g.ws_[l], 0.0f);
            fill(g.bs_[l], 0.0f);
        }

        std::vector<Matrix> dt;   // 各层误差项 dL/dz
        std::vector<Matrix> da;   // 各层激活函数的导数 f'(a)
        dt.reserve(L + 1);
        da.reserve(L + 1);
        dt.emplace_back();
        da.emplace_back();
        for (size_type l = 1; l <= L; ++l)
        {
            dt.emplace_back(1, as_[l].cols());
            da.emplace_back(1, as_[l].cols());
        }

        for (size_type sample = 0; sample < n; ++sample)
        {
            const RowView x = ti.row(sample);
            copy(input(), x);
            forward();

            const RowView y = to.row(sample);
            loss_gradient(loss_, as_[L], y, dt[L]);
            activation_derivative(output_activation_, as_[L], da[L]);
            for (size_type j = 0; j < as_[L].cols(); ++j)
            {
                dt[L](0, j) *= da[L](0, j);
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
                    const bool drop = training_ && dropout_ > 0.0f;
                    const Matrix& act = drop ? pre_dropout_act_[l - 1] : a_prev;
                    activation_derivative(hidden_activation_, act, da[l - 1]);
                    for (size_type i = 0; i < a_prev.cols(); ++i)
                    {
                        float s = 0.0f;
                        for (size_type j = 0; j < d_cur.cols(); ++j)
                        {
                            s += d_cur(0, j) * ws_[l - 1](i, j);
                        }
                        float deriv = da[l - 1](0, i);
                        if (drop)
                        {
                            deriv *= dropout_mask_[l - 1](0, i);
                        }
                        dt[l - 1](0, i) = s * deriv;
                    }
                }
            }
        }

        const float inv_n = 1.0f / static_cast<float>(n);
        for (size_type l = 0; l < L; ++l)
        {
            for (size_type i = 0; i < ws_[l].rows(); ++i)
            {
                for (size_type j = 0; j < ws_[l].cols(); ++j)
                {
                    g.ws_[l](i, j) = g.ws_[l](i, j) * inv_n + weight_decay_ * ws_[l](i, j);
                }
            }
            for (size_type j = 0; j < bs_[l].cols(); ++j)
            {
                g.bs_[l](0, j) *= inv_n;
            }
        }
    }

    void finite_diff(NN& g, float eps, const Matrix& ti, const Matrix& to)
    {
        check_shapes(ti, to);
        if (dropout_ > 0.0f)
        {
            throw std::invalid_argument("NN: finite_diff does not support dropout; use backprop");
        }
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
    void check_shapes(const Matrix& ti, const Matrix& to) const
    {
        if (ti.cols() != input().cols())
        {
            throw std::invalid_argument("NN: training input width does not match network input");
        }
        if (to.cols() != output().cols())
        {
            throw std::invalid_argument("NN: training target width does not match network output");
        }
        if (ti.rows() != to.rows())
        {
            throw std::invalid_argument("NN: training input and target counts differ");
        }
    }

    void ensure_dropout_storage()
    {
        if (dropout_mask_.size() == count_ + 1)
        {
            return;
        }
        dropout_mask_.assign(count_ + 1, Matrix());
        pre_dropout_act_.assign(count_ + 1, Matrix());
        for (size_type l = 1; l < count_; ++l)
        {
            dropout_mask_[l] = Matrix(1, as_[l].cols());
            pre_dropout_act_[l] = Matrix(1, as_[l].cols());
        }
    }

    size_type count_{0};
    std::vector<Matrix> ws_;
    std::vector<Matrix> bs_;
    std::vector<Matrix> as_;
    ActivationType hidden_activation_{ActivationType::Sigmoid};
    ActivationType output_activation_{ActivationType::Sigmoid};
    LossType loss_{LossType::MeanSquaredError};
    float weight_decay_{0.0f};
    float dropout_{0.0f};
    bool training_{false};
    std::vector<Matrix> dropout_mask_;
    std::vector<Matrix> pre_dropout_act_;
};

} // namespace nn
