#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace nn
{

struct Architecture
{
    std::size_t inputs{0};
    std::vector<std::size_t> hidden{};
    std::size_t outputs{0};

    static Architecture from_sizes(std::initializer_list<std::size_t> sizes)
    {
        if (sizes.size() < 2)
        {
            throw std::invalid_argument("Architecture needs at least input and output sizes");
        }

        Architecture arch;
        auto it = sizes.begin();
        arch.inputs = *it++;
        auto last = sizes.end();
        --last;
        for (; it != last; ++it)
        {
            arch.hidden.push_back(*it);
        }
        arch.outputs = *last;

        if (arch.inputs == 0 || arch.outputs == 0)
        {
            throw std::invalid_argument("Architecture sizes must be greater than zero");
        }
        for (std::size_t n : arch.hidden)
        {
            if (n == 0)
            {
                throw std::invalid_argument("Architecture sizes must be greater than zero");
            }
        }

        return arch;
    }

    std::vector<std::size_t> sizes() const
    {
        std::vector<std::size_t> all;
        all.reserve(hidden.size() + 2);
        all.push_back(inputs);
        all.insert(all.end(), hidden.begin(), hidden.end());
        all.push_back(outputs);
        return all;
    }
};

} // namespace nn
