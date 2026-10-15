#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>

#include "Activation.hpp"
#include "Loss.hpp"
#include "NN.hpp"
#include "Serialization.hpp"

// 以线性代数运算为基础的 ML 框架模型

namespace
{

constexpr float EPS = 1e-3f;
constexpr float RATE = 5e-1f;

using Sample = std::array<float, 3>;

[[maybe_unused]]
constexpr std::array<Sample, 4> td_xor = {{{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}};

[[maybe_unused]]
constexpr std::array<Sample, 4> td_or = {{{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}}};

[[maybe_unused]]
constexpr std::array<Sample, 4> td_and = {{{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}}};

constexpr const std::array<Sample, 4>& td = td_xor;

} // namespace

int main()
{
    std::ios::sync_with_stdio(false);

    nn::seed_rng(static_cast<std::uint32_t>(std::random_device{}()));

    const std::size_t n = td.size();
    nn::Matrix ti(n, 2);
    nn::Matrix to(n, 1);
    for (std::size_t i = 0; i < n; ++i)
    {
        ti(i, 0) = td[i][0];
        ti(i, 1) = td[i][1];
        to(i, 0) = td[i][2];
    }

    nn::Architecture arch{.inputs = 2, .hidden = {2}, .outputs = 1};
    nn::NN nn(arch);
    nn::NN g(arch);

    nn.randomize(0, 1);

    for (std::size_t i = 0; i < 1'000'000; ++i)
    {
        nn.finite_diff(g, EPS, ti, to);
        nn.learn(g, RATE);
        std::cout << std::format("cost = {:f}\n", nn.cost(ti, to));
    }

    std::cout << "------------------------------------\n";
    nn.print("nn");
    g.print("g");
    std::cout << "------------------------------------\n";
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            nn.input()(0, 0) = static_cast<float>(i);
            nn.input()(0, 1) = static_cast<float>(j);
            nn.forward();
            float y = nn.output()(0, 0);

            std::cout << std::format("{} ^ {} = {:f}\n", i, j, y);
        }
    }

    nn::save(nn, "trained.dat");

    return 0;
}
