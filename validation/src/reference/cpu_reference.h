#pragma once

/// CPU FP32 reference implementations.
/// Plain C++, no CUDA dependency — safe to compile on any machine.

/// Root-mean-square normalization: out[i] = x[i] / sqrt(var + eps) * gamma[i]
/// where var = (1/d) * sum_j x[j]^2.
void rmsnorm_ref(const float* x, const float* gamma, float* out, int d,
                 float eps = 1e-6f);

/// Matrix-vector multiply (row-major W): out[i] = sum_j W[i*d_in + j] * x[j].
void gemv_ref(const float* W, const float* x, float* out, int d_out, int d_in);

/// SiLU activation: x * sigmoid(x).
float silu_ref(float x);

/// SwiGLU: element-wise silu(g[i]) * u[i].
void swiglu_ref(const float* g, const float* u, float* out, int n);

/// Row-wise softmax (each head over seq_len tokens).
/// scores: [n_heads * seq_len] flattened.
void softmax_ref(const float* scores, float* probs, int n_heads, int seq_len);

/// Scaled dot-product attention.
/// q:   [n_heads * head_dim]          (single query vector per head)
/// K:   [n_heads * seq_len * head_dim] (row-major per head)
/// V:   [n_heads * seq_len * head_dim] (row-major per head)
/// out: [n_heads * head_dim]
void attention_ref(const float* q, const float* K, const float* V, float* out,
                   int n_heads, int seq_len, int head_dim);
