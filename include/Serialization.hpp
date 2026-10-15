#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "NN.hpp"

/*
 * tinyNN 持久化格式
 *
 *   u64  file_size (LE, 含自身)
 *   box META  [u32 size | 'META']
 *       u16 version
 *       <dtype>                   # 'FP32' / 'FP16' / 'FP8 '（4 字节字符串）
 *       box ARCH  [u32 size | 'ARCH']
 *           u32 inputs
 *           u32 outputs
 *           u32 hidden_count
 *           u32 hidden[hidden_count]
 *           <strategy> hidden_activation
 *           <strategy> output_activation
 *           <strategy> loss
 *   box WGHT  [u32 size | 'WGHT']
 *       f32 data[...]             # 按层拼接
 *   box BIAS  [u32 size | 'BIAS']
 *       f32 data[...]             # 按层拼接
 *
 * <strategy> = u8 len + ascii 名称 + 奇数补 0（字段总长恒为偶数）
 * 所有数值（块头/整数/浮点）均为 little-endian。
 */

namespace nn
{

using Bytes = std::vector<std::byte>;

namespace detail
{

inline void put_u8(Bytes& b, std::uint8_t v)
{
    b.push_back(static_cast<std::byte>(v));
}

inline void put_u16_le(Bytes& b, std::uint16_t v)
{
    put_u8(b, static_cast<std::uint8_t>(v));
    put_u8(b, static_cast<std::uint8_t>(v >> 8));
}

inline void put_u32_le(Bytes& b, std::uint32_t v)
{
    put_u8(b, static_cast<std::uint8_t>(v));
    put_u8(b, static_cast<std::uint8_t>(v >> 8));
    put_u8(b, static_cast<std::uint8_t>(v >> 16));
    put_u8(b, static_cast<std::uint8_t>(v >> 24));
}

inline void put_u64_le(Bytes& b, std::uint64_t v)
{
    put_u32_le(b, static_cast<std::uint32_t>(v));
    put_u32_le(b, static_cast<std::uint32_t>(v >> 32));
}

inline void put_f32_le(Bytes& b, float f)
{
    put_u32_le(b, std::bit_cast<std::uint32_t>(f));
}

inline void put_fourcc(Bytes& b, std::string_view cc)
{
    if (cc.size() != 4)
    {
        throw std::invalid_argument("tinyNN: fourcc must be exactly 4 bytes");
    }
    for (char c : cc)
    {
        put_u8(b, static_cast<std::uint8_t>(static_cast<unsigned char>(c)));
    }
}

inline void put_strategy(Bytes& b, std::string_view name)
{
    if (name.size() > 255)
    {
        throw std::invalid_argument("tinyNN: strategy name too long");
    }
    put_u8(b, static_cast<std::uint8_t>(name.size()));
    for (char c : name)
    {
        put_u8(b, static_cast<std::uint8_t>(static_cast<unsigned char>(c)));
    }
    if (((1 + name.size()) % 2) != 0)
    {
        put_u8(b, 0);
    }
}

class Reader
{
public:
    explicit Reader(std::span<const std::byte> data) noexcept
        : data_(data) {}

    std::size_t remaining() const noexcept { return data_.size() - pos_; }
    bool empty() const noexcept { return pos_ == data_.size(); }

    std::uint8_t u8()
    {
        need(1);
        return std::to_integer<std::uint8_t>(data_[pos_++]);
    }

    std::uint16_t u16_le()
    {
        const std::uint16_t lo = u8();
        const std::uint16_t hi = u8();
        return static_cast<std::uint16_t>((hi << 8) | lo);
    }

    std::uint32_t u32_le()
    {
        const std::uint32_t b0 = u8();
        const std::uint32_t b1 = u8();
        const std::uint32_t b2 = u8();
        const std::uint32_t b3 = u8();
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    std::uint64_t u64_le()
    {
        const std::uint64_t lo = u32_le();
        const std::uint64_t hi = u32_le();
        return lo | (hi << 32);
    }

    float f32_le()
    {
        return std::bit_cast<float>(u32_le());
    }

    Reader sub(std::size_t n)
    {
        need(n);
        const Reader r{data_.subspan(pos_, n)};
        pos_ += n;
        return r;
    }

private:
    void need(std::size_t n) const
    {
        if (n > data_.size() - pos_)
        {
            throw std::runtime_error("tinyNN: truncated data");
        }
    }

    std::span<const std::byte> data_;
    std::size_t pos_{0};
};

inline std::string read_strategy(Reader& r)
{
    const std::uint8_t len = r.u8();
    std::string s;
    s.resize(len);
    for (std::size_t i = 0; i < len; ++i)
    {
        s[i] = static_cast<char>(r.u8());
    }
    if (((1 + len) % 2) != 0)
    {
        (void)r.u8();
    }
    return s;
}

inline Bytes make_box(std::string_view fourcc, const Bytes& payload)
{
    Bytes out;
    put_u32_le(out, static_cast<std::uint32_t>(8 + payload.size()));
    put_fourcc(out, fourcc);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

inline std::string read_fourcc(Reader& r)
{
    std::string cc(4, '\0');
    for (char& c : cc)
    {
        c = static_cast<char>(r.u8());
    }
    return cc;
}

inline void check_dtype(std::string_view dtype)
{
    if (dtype == "FP32")
    {
        return;
    }
    if (dtype == "FP16" || dtype == "FP8 ")
    {
        throw std::runtime_error("tinyNN: dtype '" + std::string(dtype) + "' is not implemented yet");
    }
    throw std::runtime_error("tinyNN: unknown dtype '" + std::string(dtype) + "'");
}

} // namespace detail

inline std::string_view activation_name(ActivationType t) noexcept
{
    switch (t)
    {
    case ActivationType::Sigmoid: return "sigmoid";
    case ActivationType::Tanh:    return "tanh";
    case ActivationType::Relu:    return "relu";
    }
    return "sigmoid";
}

inline ActivationType activation_from(std::string_view name)
{
    if (name == "sigmoid") return ActivationType::Sigmoid;
    if (name == "tanh")    return ActivationType::Tanh;
    if (name == "relu")    return ActivationType::Relu;
    throw std::invalid_argument("tinyNN: unknown activation '" + std::string(name) + "'");
}

inline std::string_view loss_name(LossType t) noexcept
{
    switch (t)
    {
    case LossType::MeanSquaredError:   return "mse";
    case LossType::BinaryCrossEntropy: return "bce";
    }
    return "mse";
}

inline LossType loss_from(std::string_view name)
{
    if (name == "mse") return LossType::MeanSquaredError;
    if (name == "bce") return LossType::BinaryCrossEntropy;
    throw std::invalid_argument("tinyNN: unknown loss '" + std::string(name) + "'");
}

inline Bytes serialize(const NN& model)
{
    const std::size_t L = model.count();

    std::vector<std::size_t> sizes;
    sizes.reserve(L + 1);
    for (std::size_t i = 0; i <= L; ++i)
    {
        sizes.push_back(model.activations(i).cols());
    }

    // ARCH
    Bytes arch;
    detail::put_u32_le(arch, static_cast<std::uint32_t>(sizes.front()));
    detail::put_u32_le(arch, static_cast<std::uint32_t>(sizes.back()));
    detail::put_u32_le(arch, static_cast<std::uint32_t>(sizes.size() - 2));
    for (std::size_t i = 1; i + 1 < sizes.size(); ++i)
    {
        detail::put_u32_le(arch, static_cast<std::uint32_t>(sizes[i]));
    }
    detail::put_strategy(arch, activation_name(model.hidden_activation()));
    detail::put_strategy(arch, activation_name(model.output_activation()));
    detail::put_strategy(arch, loss_name(model.loss()));

    // META
    Bytes meta;
    detail::put_u16_le(meta, 1);   // format version
    detail::put_fourcc(meta, "FP32");
    const Bytes arch_box = detail::make_box("ARCH", arch);
    meta.insert(meta.end(), arch_box.begin(), arch_box.end());

    // WGHT / BIAS（dtype 统一由 META 声明）
    Bytes wght;
    Bytes bias;
    for (std::size_t i = 0; i < L; ++i)
    {
        const Matrix& w = model.weights(i);
        for (std::size_t r = 0; r < w.rows(); ++r)
        {
            for (std::size_t c = 0; c < w.cols(); ++c)
            {
                detail::put_f32_le(wght, w(r, c));
            }
        }
        const Matrix& b = model.biases(i);
        for (std::size_t r = 0; r < b.rows(); ++r)
        {
            for (std::size_t c = 0; c < b.cols(); ++c)
            {
                detail::put_f32_le(bias, b(r, c));
            }
        }
    }

    // assemble
    Bytes out;
    detail::put_u64_le(out, 0);    // placeholder for file size
    for (const Bytes& box : {detail::make_box("META", meta),
                             detail::make_box("WGHT", wght),
                             detail::make_box("BIAS", bias)})
    {
        out.insert(out.end(), box.begin(), box.end());
    }

    const std::uint64_t total = out.size();
    for (std::size_t i = 0; i < 8; ++i)
    {
        out[i] = static_cast<std::byte>((total >> (8 * i)) & 0xFF);
    }
    return out;
}

inline NN read_meta(detail::Reader mr)
{
    const std::uint16_t version = mr.u16_le();
    if (version != 1)
    {
        throw std::runtime_error("tinyNN: unsupported format version");
    }
    detail::check_dtype(detail::read_fourcc(mr));

    const std::uint32_t asize = mr.u32_le();
    if (asize < 8)
    {
        throw std::runtime_error("tinyNN: invalid ARCH box size");
    }
    if (detail::read_fourcc(mr) != "ARCH")
    {
        throw std::runtime_error("tinyNN: expected ARCH box inside META");
    }

    detail::Reader ar = mr.sub(asize - 8);
    const std::uint32_t inputs = ar.u32_le();
    const std::uint32_t outputs = ar.u32_le();
    const std::uint32_t hidden_count = ar.u32_le();

    std::vector<std::size_t> hidden;
    hidden.reserve(hidden_count);
    for (std::uint32_t i = 0; i < hidden_count; ++i)
    {
        hidden.push_back(ar.u32_le());
    }

    const ActivationType hidden_act = activation_from(detail::read_strategy(ar));
    const ActivationType output_act = activation_from(detail::read_strategy(ar));
    const LossType loss = loss_from(detail::read_strategy(ar));

    Architecture arch;
    arch.inputs = inputs;
    arch.hidden = std::move(hidden);
    arch.outputs = outputs;

    NN model(arch);
    model.set_hidden_activation(hidden_act);
    model.set_output_activation(output_act);
    model.set_loss(loss);
    return model;
}

inline void read_matrix_block(NN& model, bool is_weights, detail::Reader br)
{
    for (std::size_t i = 0; i < model.count(); ++i)
    {
        Matrix& m = is_weights ? model.weights(i) : model.biases(i);
        for (std::size_t r = 0; r < m.rows(); ++r)
        {
            for (std::size_t c = 0; c < m.cols(); ++c)
            {
                m(r, c) = br.f32_le();
            }
        }
    }
}

inline NN deserialize(std::span<const std::byte> data)
{
    detail::Reader r{data};
    const std::uint64_t file_size = r.u64_le();
    if (file_size != data.size())
    {
        throw std::runtime_error("tinyNN: file size field does not match actual size");
    }

    NN model({1, 1});
    bool have_model = false;

    while (!r.empty())
    {
        const std::uint32_t size = r.u32_le();
        if (size < 8)
        {
            throw std::runtime_error("tinyNN: invalid box size");
        }
        const std::string fourcc = detail::read_fourcc(r);
        detail::Reader body = r.sub(size - 8);   // unknown 块：构造即消费，自动跳过

        if (fourcc == "META")
        {
            model = read_meta(body);
            have_model = true;
        }
        else if (fourcc == "WGHT")
        {
            if (!have_model)
            {
                throw std::runtime_error("tinyNN: WGHT box before META box");
            }
            read_matrix_block(model, true, body);
        }
        else if (fourcc == "BIAS")
        {
            if (!have_model)
            {
                throw std::runtime_error("tinyNN: BIAS box before META box");
            }
            read_matrix_block(model, false, body);
        }
    }

    if (!have_model)
    {
        throw std::runtime_error("tinyNN: missing META box");
    }
    return model;
}

inline void save(const NN& model, const std::string& path)
{
    const Bytes bytes = serialize(model);
    std::ofstream f(path, std::ios::binary);
    if (!f)
    {
        throw std::runtime_error("tinyNN: cannot open '" + path + "' for writing");
    }
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (!f)
    {
        throw std::runtime_error("tinyNN: write failed for '" + path + "'");
    }
}

inline NN load(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        throw std::runtime_error("tinyNN: cannot open '" + path + "' for reading");
    }
    const std::streamsize n = f.tellg();
    if (n < 0)
    {
        throw std::runtime_error("tinyNN: cannot determine size of '" + path + "'");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(n));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(bytes.data()), n);
    if (!f)
    {
        throw std::runtime_error("tinyNN: read failed for '" + path + "'");
    }
    return deserialize(bytes);
}

} // namespace nn
