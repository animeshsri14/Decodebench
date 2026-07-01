# DecodeBench - References

All papers and technical documents cited in the DecodeBench work.

---

## Core / Foundational

1. Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM, 52(4), 65–76.

2. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.

3. Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024.

4. Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* NeurIPS 2024.

5. Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.

6. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.

---

## Quantization

7. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022.

8. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.

9. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323. (ICLR 2023.)

10. Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024.

---

## Megakernel / Fusion Systems

11. Spector, B., Juravsky, J., Sul, S., Dugan, O., Lim, D., Fu, D., Arora, S., & Ré, C. (2025). *Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B.* Stanford / Hazy Research blog post, May 27 2025. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

12. Jia, Z., et al. (2025). *Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs.* arXiv:2512.22219.

13. *ClusterFusion* (2025). Hopper cluster-level primitive fusion for decode. arXiv:2508.18850.

14. *ClusterFusion++* (2026). Full transformer-block cluster fusion with CUDA-Graph-compatible mode. arXiv:2604.23553.

15. *ETC / Event Tensor Compiler* (2026). *Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernels.* MLSys 2026. arXiv:2604.13327.

16. *Ada-MK* (2026). *Adaptive MegaKernel Optimization via Automated DAG-based Search.* arXiv:2605.11581.

17. Jaber, J., & Jaber, O. (2026). *AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis.* RightNow AI. arXiv:2606.09682. Submitted June 8 2026.

---

## Related Inference / Profiling Work

18. *SPEED-Bench* (2026). *A Unified and Diverse Benchmark for Speculative Decoding.* NVIDIA. ICML 2026. arXiv:2604.09557.

19. *FlashSampling* (2026). Fuses exact sampling into the LM-head matmul; logits never hit HBM. arXiv:2603.15854.

20. *SKIP / TKLQT* (2025). Profiler and metric locating the CPU-bound ↔ GPU-bound transition; notes fusion helps launch-bound regimes. arXiv:2504.11750.

21. *Chopper* (2025). Multi-level GPU characterization for LLM training; ranks DVFS above launch overhead. arXiv:2512.08242.

22. *Hybrid JIT-CUDA Graph* (2026). Splits inference into graph-replayed static + JIT dynamic regions. arXiv:2604.23467.

23. Ye, Z., Chen, L., Lai, R., Lin, W., Zhang, Y., Wang, S., Chen, T., Kasikci, B., Grover, V., & Krishnamurthy, A. (2025). *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.* arXiv:2501.01005. MLSys 2025.

24. Park, S., Jeon, S., Lee, C., Jeon, S., Kim, B.-S., & Lee, J. (2025). *A Survey on Inference Engines for Large Language Models: Perspectives on Optimization and Efficiency.* arXiv:2505.01658. ACM Transactions on Intelligent Systems and Technology, 2026.

---

## NVIDIA Technical Documents

25. NVIDIA (2020). *NVIDIA A100 Tensor Core GPU Architecture White Paper.*

26. NVIDIA (2022). *NVIDIA H100 Tensor Core GPU Architecture White Paper.*

27. NVIDIA (2023). *Nsight Compute User Guide.* https://docs.nvidia.com/nsight-compute/

28. NVIDIA. *CUDA C++ Programming Guide §3.2.8 - CUDA Graphs.* https://docs.nvidia.com/cuda/cuda-c-programming-guide/

