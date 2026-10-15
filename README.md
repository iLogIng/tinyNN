# tinyNN

一个以线性代数运算为基础、用 C++20 实现的极简神经网络库。整个库为 header-only，`main.cpp` 演示用有限差分法训练一个 XOR 网络。

## 特性

- **header-only**：所有类以单头文件形式放在 `include/`，无独立编译单元。
- **函数式算子**：矩阵运算以自由函数提供（`dot`/`sum`/`fill`/`randomize`/`copy`/`print`），元素访问保留 `operator()`。
- **行视图**：`RowView = std::span<const float>`，非拥有、零开销。
- **架构配置**：`Architecture{inputs, hidden, outputs}` 描述网络结构。
- **可替换激活/损失**：隐藏层与输出层激活分别可配（`Sigmoid`/`Tanh`/`Relu`），损失可选（`MSE`/`BinaryCrossEntropy`）。
- **RAII**：`Matrix` 以 `std::vector<float>` 持有数据，`NN` 以 `std::vector<Matrix>` 持有各层，无需手动释放。

## 目录结构

```
.
├── CMakeLists.txt
├── main.cpp              # XOR 训练示例
├── NN.h                  # 迁移前的 C 实现，仅作参考
└── include/
    ├── Architecture.hpp  # 网络结构配置
    ├── Util.hpp          # 随机数引擎 rng()/seed_rng()
    ├── Activation.hpp    # 激活策略 Sigmoid/Tanh/Relu + 枚举分派
    ├── Loss.hpp          # 损失策略 MSE/BinaryCrossEntropy + 枚举分派
    ├── Matrix.hpp        # Matrix / RowView / 矩阵算子
    └── NN.hpp            # NN 类（前向、代价、反向传播、有限差分、学习）
```

## 构建与运行

> >=C++20  
> GCC 13+ / Clang 17+  
> CMake 3.20+

```sh
cmake -S . -B build
cmake --build build -j
./build/main
```

`build/compile_commands.json` 由CMake自动生成

## 使用示例

```cpp
#include "NN.hpp"

nn::Architecture arch{.inputs = 2, .hidden = {2}, .outputs = 1};
nn::NN nn(arch);
// 扁平方式构造
// nn::NN nn({2, 2, 1});
nn::NN g(arch);                       // 梯度缓冲

nn.randomize(0.0f, 1.0f);
// ... 准备训练矩阵 ti (n×2) 与 to (n×1) ...

for (int i = 0; i < 1000000; ++i) {
    nn.finite_diff(g, 1e-3f, ti, to); // 数值梯度
    nn.learn(g, 5e-1f);               // 梯度下降
}

nn.input()(0, 0) = 1.0f;              // 写入输入
nn.input()(0, 1) = 1.0f;
nn.forward();
float y = nn.output()(0, 0);         // 读取输出
```

## 说明

- 默认激活为 `Sigmoid`（隐藏层与输出层），默认损失为 `MSE`；可通过 `set_hidden_activation`/`set_output_activation`/`set_loss` 更换。
- 训练支持数值梯度（`finite_diff`）与解析反向传播（`backprop`），两者都与所选激活/损失保持一致。
- `BinaryCrossEntropy` 期望输出层为 sigmoid。
- `Architecture::from_sizes` 会校验尺寸（至少输入+输出且均大于 0），非法输入抛 `std::invalid_argument`；`cost`/`backprop`/`finite_diff` 会校验训练数据维度。
- 矩阵的 `stride` 恒等于 `cols`，因此行连续，可用 `std::span` 表达。


