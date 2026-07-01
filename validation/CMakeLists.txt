cmake_minimum_required(VERSION 3.22)
project(decodebench_val LANGUAGES CXX)

# ---- options -----------------------------------------------------------
option(DECODEBENCH_CPU_ONLY "Build CPU-only tests without CUDA" OFF)

# ---- detect CUDA -------------------------------------------------------
set(DECODEBENCH_HAS_CUDA OFF)
if(NOT DECODEBENCH_CPU_ONLY)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(DECODEBENCH_HAS_CUDA ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;89;90" CACHE STRING
                "CUDA architectures (semicolon-separated). Override with -DCMAKE_CUDA_ARCHITECTURES=61 for GTX 1060, etc.")
        endif()
        message(STATUS "CUDA found — building GPU targets for arch: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "CUDA compiler not found — building CPU-only")
    endif()
else()
    message(STATUS "CUDA disabled (DECODEBENCH_CPU_ONLY=ON) — building CPU-only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- include directory -------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- CPU reference library (always built) ------------------------------
add_library(cpu_reference STATIC
    src/reference/cpu_reference.cpp
)
target_include_directories(cpu_reference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reference
)

# ---- test_reference (always built) -------------------------------------
add_executable(test_reference
    src/tests/test_reference.cpp
)
target_link_libraries(test_reference PRIVATE cpu_reference)

# ---- testing -----------------------------------------------------------
enable_testing()
add_test(NAME test_reference COMMAND test_reference)

# ---- GPU targets (only when CUDA is available) -------------------------
if(DECODEBENCH_HAS_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # GPU kernel library — includes device headers as source dependencies
    add_library(decodebench_kernels STATIC
        src/kernels/unfused/gemv.cu
        src/kernels/unfused/rmsnorm.cu
        src/kernels/unfused/swiglu.cu
        src/kernels/unfused/attn_scores.cu
        src/kernels/unfused/softmax.cu
        src/kernels/unfused/attn_v.cu
        src/kernels/fused/f1.cu
        src/kernels/fused/f2.cu
        src/kernels/fused/f4.cu
    )
    target_include_directories(decodebench_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/device
    )

    # bench_variant — timing harness (uses cuBLAS for G2 reference only)
    add_executable(bench_variant
        src/bench/bench_variant.cu
    )
    target_link_libraries(bench_variant PRIVATE decodebench_kernels cublas stdc++ m)

    # calibrate — null launch + G2 gate calibration
    add_executable(calibrate
        src/bench/calibrate.cu
    )
    target_link_libraries(calibrate PRIVATE decodebench_kernels cublas stdc++)
endif()
