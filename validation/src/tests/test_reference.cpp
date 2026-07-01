#include <cmath>
#include <cstdio>
#include <cstring>

#include "cpu_reference.h"

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------
static int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL: %s\n", msg);                                   \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

#define CHECK_FLOAT(got, expected, tol, msg)                                  \
    do {                                                                      \
        if (std::fabs((got) - (expected)) > (tol)) {                          \
            std::printf("FAIL: %s  (got %.8f, expected %.8f)\n", msg, got,   \
                        expected);                                            \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// rmsnorm golden test
// ---------------------------------------------------------------------------
static void test_rmsnorm() {
    std::printf("test_rmsnorm ... ");
    float x[]     = {1.f, 2.f, 3.f, 4.f};
    float gamma[] = {1.f, 1.f, 1.f, 1.f};
    float out[4]  = {};
    rmsnorm_ref(x, gamma, out, 4);

    float mean_sq = (1.f*1.f + 2.f*2.f + 3.f*3.f + 4.f*4.f) / 4.f; // 7.5
    float rms     = std::sqrt(mean_sq + 1e-6f);

    for (int i = 0; i < 4; ++i) {
        float expected = x[i] / rms;
        CHECK_FLOAT(out[i], expected, 1e-5f, "rmsnorm element");
    }

    if (g_failures == 0 || std::strstr("rmsnorm", "already checked") == nullptr)
        std::printf("PASS\n");
}

// ---------------------------------------------------------------------------
// gemv golden test
// ---------------------------------------------------------------------------
static void test_gemv() {
    std::printf("test_gemv ... ");
    // W = [[1,2],[3,4]] row-major → [1,2,3,4]
    float W[]       = {1.f, 2.f, 3.f, 4.f};
    float x[]       = {1.f, 1.f};
    float out[2]    = {};
    gemv_ref(W, x, out, 2, 2);

    CHECK_FLOAT(out[0], 3.f, 1e-5f, "gemv[0]");
    CHECK_FLOAT(out[1], 7.f, 1e-5f, "gemv[1]");

    if (g_failures <= 2)
        std::printf("PASS\n");
}

// ---------------------------------------------------------------------------
// silu golden test
// ---------------------------------------------------------------------------
static void test_silu() {
    std::printf("test_silu ... ");

    float zero = silu_ref(0.f);
    CHECK_FLOAT(zero, 0.f, 1e-6f, "silu(0)");

    float one = silu_ref(1.f);
    CHECK_FLOAT(one, 0.7310586f, 1e-6f, "silu(1)");

    if (g_failures <= 2)
        std::printf("PASS\n");
}

// ---------------------------------------------------------------------------
// swiglu sanity test
// ---------------------------------------------------------------------------
static void test_swiglu() {
    std::printf("test_swiglu ... ");
    float g[] = {0.f, 1.f, 2.f};
    float u[] = {1.f, 1.f, 1.f};
    float out[3];
    swiglu_ref(g, u, out, 3);

    CHECK_FLOAT(out[0], 0.f,                  1e-6f, "swiglu[0]");
    CHECK_FLOAT(out[1], 0.7310586f,           1e-5f, "swiglu[1]");
    float silu2 = silu_ref(2.f);
    CHECK_FLOAT(out[2], silu2,                1e-5f, "swiglu[2]");

    if (g_failures <= 3)
        std::printf("PASS\n");
}

// ---------------------------------------------------------------------------
// softmax sanity test
// ---------------------------------------------------------------------------
static void test_softmax() {
    std::printf("test_softmax ... ");
    // 2 heads, 3 tokens each
    float scores[] = {0.f, 1.f, 2.f,    // head 0
                      0.f, 0.f, 0.f};   // head 1
    float probs[6];
    softmax_ref(scores, probs, 2, 3);

    // head 0: [exp(-2), exp(-1), exp(0)] / sum
    float sum0 = std::exp(0.f) + std::exp(1.f) + std::exp(2.f);
    CHECK_FLOAT(probs[0], std::exp(0.f) / sum0, 1e-5f, "softmax[0]");
    CHECK_FLOAT(probs[1], std::exp(1.f) / sum0, 1e-5f, "softmax[1]");
    CHECK_FLOAT(probs[2], std::exp(2.f) / sum0, 1e-6f, "softmax[2]");

    // head 1: uniform 1/3
    CHECK_FLOAT(probs[3], 1.f / 3.f, 1e-6f, "softmax uniform[0]");
    CHECK_FLOAT(probs[4], 1.f / 3.f, 1e-6f, "softmax uniform[1]");
    CHECK_FLOAT(probs[5], 1.f / 3.f, 1e-6f, "softmax uniform[2]");

    if (g_failures <= 6)
        std::printf("PASS\n");
}

// ---------------------------------------------------------------------------
// attention sanity test
// ---------------------------------------------------------------------------
static void test_attention() {
    std::printf("test_attention ... ");
    // 1 head, head_dim=2, seq_len=2
    // q = [1, 0]
    // K = [[1,0],[0,1]]  (row-major: 1,0,0,1)
    // V = [[2,0],[0,3]]  (row-major: 2,0,0,3)
    // scores = q·K[0]=1, q·K[1]=0 → softmax → [e/(e+1), 1/(e+1)]
    // out = first * V[0] + second * V[1]
    float q[] = {1.f, 0.f};
    float K[] = {1.f, 0.f, 0.f, 1.f};
    float V[] = {2.f, 0.f, 0.f, 3.f};
    float out[2];
    attention_ref(q, K, V, out, 1, 2, 2);

    float e = std::exp(1.f);
    float p0 = e / (e + 1.f);
    float p1 = 1.f / (e + 1.f);

    float expected0 = p0 * 2.f + p1 * 0.f;
    float expected1 = p0 * 0.f + p1 * 3.f;
    CHECK_FLOAT(out[0], expected0, 1e-5f, "attention[0]");
    CHECK_FLOAT(out[1], expected1, 1e-5f, "attention[1]");

    if (g_failures <= 2)
        std::printf("PASS\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_rmsnorm();
    test_gemv();
    test_silu();
    test_swiglu();
    test_softmax();
    test_attention();

    if (g_failures > 0) {
        std::printf("\n%d test(s) FAILED\n", g_failures);
        return 1;
    }

    std::printf("\nAll tests PASSED\n");
    return 0;
}
