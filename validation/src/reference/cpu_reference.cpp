#include "cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>

void rmsnorm_ref(const float* x, const float* gamma, float* out, int d,
                 float eps) {
    // var = (1/d) * sum_j x[j]^2
    float sum_sq = 0.f;
    for (int i = 0; i < d; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(d) + eps);
    for (int i = 0; i < d; ++i) {
        out[i] = x[i] / rms * gamma[i];
    }
}

// row-major W
void gemv_ref(const float* W, const float* x, float* out, int d_out, int d_in) {
    for (int i = 0; i < d_out; ++i) {
        float acc = 0.f;
        for (int j = 0; j < d_in; ++j) {
            acc += W[i * d_in + j] * x[j];
        }
        out[i] = acc;
    }
}

float silu_ref(float x) {
    return x / (1.f + std::exp(-x)); // x * sigmoid(x)
}

void swiglu_ref(const float* g, const float* u, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = silu_ref(g[i]) * u[i];
    }
}

// row-wise over seq_len
void softmax_ref(const float* scores, float* probs, int n_heads, int seq_len) {
    for (int h = 0; h < n_heads; ++h) {
        const float* row = scores + h * seq_len;
        float*       prow = probs + h * seq_len;

        // subtract max for numerical stability
        float mx = row[0];
        for (int i = 1; i < seq_len; ++i) {
            if (row[i] > mx) mx = row[i];
        }

        float sum = 0.f;
        for (int i = 0; i < seq_len; ++i) {
            prow[i] = std::exp(row[i] - mx);
            sum += prow[i];
        }
        for (int i = 0; i < seq_len; ++i) {
            prow[i] /= sum;
        }
    }
}

void attention_ref(const float* q, const float* K, const float* V, float* out,
                   int n_heads, int seq_len, int head_dim) {
    for (int h = 0; h < n_heads; ++h) {
        const float* qh = q + h * head_dim;
        const float* Kh = K + h * seq_len * head_dim;
        const float* Vh = V + h * seq_len * head_dim;
        float*       oh = out + h * head_dim;

        // 1. Compute dot-product scores: scores[t] = dot(qh, Kh[t])
        float* scores = new float[seq_len];
        for (int t = 0; t < seq_len; ++t) {
            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d) {
                dot += qh[d] * Kh[t * head_dim + d];
            }
            scores[t] = dot;
        }

        // 2. Softmax over scores
        float* probs = new float[seq_len];
        {
            float mx = scores[0];
            for (int i = 1; i < seq_len; ++i) {
                if (scores[i] > mx) mx = scores[i];
            }
            float sum = 0.f;
            for (int i = 0; i < seq_len; ++i) {
                probs[i] = std::exp(scores[i] - mx);
                sum += probs[i];
            }
            for (int i = 0; i < seq_len; ++i) {
                probs[i] /= sum;
            }
        }

        // 3. Weighted sum over V: out[d] = sum_t probs[t] * Vh[t][d]
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.f;
            for (int t = 0; t < seq_len; ++t) {
                acc += probs[t] * Vh[t * head_dim + d];
            }
            oh[d] = acc;
        }

        delete[] scores;
        delete[] probs;
    }
}
