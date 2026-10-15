#ifdef __cplusplus
extern "C" {
#endif

#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(xs) (sizeof((xs)) / sizeof((xs)[0]))

/** 
 * Neuro Network
 * 1. 线性代数计算方法
 * 2. 前向传播计算
 * 3. 激活函数
 * 4. 以上各个算子的运算组合
*/

/* 矩阵 */
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

float rand_float();
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float val);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_print(Mat mat, const char* name, size_t padding);

#define MAT_PRINT(m) mat_print(m, #m, 0)

/* 神经网络结构 */
typedef struct {
    /* 记录网络层数 */
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count+1
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
void nn_rand(NN nn, float low, float high);
#define NN_PRINT(nn) nn_print(nn, #nn)
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);

#endif // NN_H

#ifndef NN_IMPLEMENTATION
#define NN_IMPLEMENTATION

float rand_float()
{
    return rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = (float*)NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_fill(Mat m, float val)
{
    assert(m.es != NULL);
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MAT_AT(m, i, j) = val;
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);
    for (size_t i = 0; i < dst.rows; ++i)
    {
        for (size_t j = 0; j < dst.cols; ++j)
        {
            MAT_AT(dst, i, j) = 0.f;
            for (size_t k = 0; k < a.cols; ++k)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
    
}

void mat_copy(Mat dst, Mat src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; ++i)
    {
        for (size_t j = 0; j < dst.cols; ++j)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i)
    {
        for (size_t j = 0; j < dst.cols; ++j)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void mat_print(Mat m, const char* name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; ++i)
    {
        printf("    ");
        for (size_t j = 0; j < m.cols; ++j)
        {
            printf("%*s%f", (int)padding, "", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

/**
 * size_t layers[] = {2, 2, 1} // 数量(2)，隐藏层[idx]神经元数量(2)，输出数量(1)
 * layers[] = {
 *      size_t inputs;
 *      size_t *layer_neuros;
 *      size_t outputs;
 * }
 * NN nn = nn_alloc(layers, ARRAY_LEN(layers))
 */

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count >= 1);

    NN nn;
    nn.count = arch_count - 1; // 减去输入层

    nn.ws = (Mat*)NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = (Mat*)NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = (Mat*)NN_MALLOC(sizeof(*nn.as) * (nn.count + 1)); // 包含一个输入层
    NN_ASSERT(nn.as != NULL);

    /* 输入参数数量 */
    nn.as[0] = mat_alloc(1, arch[0]);    
    for (size_t i = 1; i < arch_count; ++i)
    {
        /* 每层的权重组数 == 该层的神经元数量 */
        /* 每组权重的数量 == 该层的参数数量 */
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        /* 每层的偏置数量 == 该层的神经元数量 */
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        /* 每层都输出数量 == 该层的神经元数量 */
        /* 1 neuro : 1 output */
        nn.as[i] = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; ++i)
    {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

void nn_backprop(NN nn, Mat ti, Mat to, NN g) {
    size_t n = ti.rows;     // 样本数量
    size_t L = nn.count;    // 层数

    // 清空梯度
    for (size_t l = 0; l < L; ++l) {
        mat_fill(g.ws[l], 0.0f);
        mat_fill(g.bs[l], 0.0f);
    }

    // 为每一层分配误差项
    Mat *dt = (Mat*)malloc(sizeof(Mat) * (L + 1)); // dt[1..L]有效
    for (size_t l = 1; l <= L; ++l) {
        dt[l] = mat_alloc(1, nn.as[l].cols);
    }

    // 遍历每个样本
    for (size_t sample = 0; sample < n; ++sample) {
        // 前向传播
        // 得到最终结果以及每层的计算结果
        Mat x = mat_row(ti, sample);
        mat_copy(NN_INPUT(nn),x);
        nn_forward(nn);

        // 输出层误差
        Mat y = mat_row(to, sample);
        for (size_t j = 0; j < nn.as[L].cols; ++j) {
            float out = MAT_AT(nn.as[L], 0, j);
            float target = MAT_AT(y, 0, j);
            float grad = 2.0f * (out - target) * out * (1.0f - out);
            MAT_AT(dt[L], 0, j) = grad;
        }

        // dLoss / dw = dLoss/dan * dan/dzn * dzn/da{n-1} * da{n-1}/dz{n-1} * dz{n-1}/da{n-2} * da{n-2}/dz{n-2} * ... * dz2/da1 * da1/dz1 * dz1/dw1
        // 以上
        // z是线性组合
        // a是各层激活函数的结果
        // w是各层参数
        // 反向传播，计算各层误差并累加梯度
        for (size_t l = L; l >= 1; --l) {
            // 当前层的权重与偏置的梯度
            Mat a_prev = nn.as[l - 1];
            Mat d_cur = dt[l];
            
            // w 权重梯度 (a_prev)^T * d_cur
            for (size_t i = 0; i < a_prev.cols; ++i) {
                for (size_t j = 0; j < d_cur.cols; ++j) {
                    MAT_AT(g.ws[l - 1], i, j) += MAT_AT(a_prev, 0, i) * MAT_AT(d_cur, 0, j);
                }
            }
            // b 偏置梯度 d_cur
            for (size_t j = 0; j < d_cur.cols; ++j) {
                MAT_AT(g.bs[l - 1], 0, j) += MAT_AT(d_cur, 0, j);                
            }

            // 若前一层存在，计算 dt[l - 1]
            if (l > 1) {
                // dt[l - 1] = (d_cur * W^T) * a_prev * (1 - a_prev)
                for (size_t i = 0; i < a_prev.cols; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < d_cur.cols; ++j) {
                        sum += MAT_AT(d_cur, 0, j) * MAT_AT(nn.ws[l - 1], i, j);
                    }
                    float a = MAT_AT(a_prev, 0, i);
                    MAT_AT(dt[l - 1], 0, i) = sum * a * (1.0f - a);
                }
            }
        }
    }

    // 对累加的梯度取平均
    for (size_t l = 0; l < L; ++l) {
        for (size_t i = 0; i < nn.ws[l].rows; ++i) {
            for (size_t j = 0; j < nn.ws[l].cols; ++j) {
                MAT_AT(g.ws[l], i, j) /= (float)n;
            }
        }
        for (size_t j = 0; j < nn.bs[l].cols; ++j) {
            MAT_AT(g.bs[l], 0, j) /= (float)n;
        }
    }

    // 释放临时dt矩阵
    for (size_t l = 1; l <= L; ++l) {
        free(dt[l].es);
    }
    free(dt);
}

float nn_cost(NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    float c = 0.f;
    for (size_t i = 0; i < n; ++i)
    {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        for (size_t j = 0; j < to.cols; ++j)
        {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d * d;
        }
    }

    return c / n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
    float saved;
    float c = nn_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; ++i)
    {
        for (size_t j = 0; j < nn.ws[i].rows; ++j)
        {
            for (size_t k = 0; k < nn.ws[i].cols; ++k)
            {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j)
        {
            for (size_t k = 0; k < nn.bs[i].cols; ++k)
            {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.count; ++i)
    {
        for (size_t j = 0; j < nn.ws[i].rows; ++j)
        {
            for (size_t k = 0; k < nn.ws[i].cols; ++k)
            {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }
        for (size_t j = 0; j < nn.bs[i].rows; ++j)
        {
            for (size_t k = 0; k < nn.bs[i].cols; ++k)
            {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

void nn_free(NN nn)
{
    for (size_t i = 0; i <= nn.count; ++i) {
        free(nn.as[i].es);
    }
    for (size_t i = 0; i < nn.count; ++i) {
        free(nn.ws[i].es);
        free(nn.bs[i].es);
    }
    free(nn.as);
    free(nn.bs);
    free(nn.ws);
}


#endif // NN_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

