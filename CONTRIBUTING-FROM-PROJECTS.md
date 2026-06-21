# Hello GPU 贡献指南：致相关项目作者

本文档面向以下六个项目的作者和维护者，说明你们的项目中哪些内容可以直接转化为 Hello GPU 教程的章节素材。

> Hello GPU 是一本面向 AI Infra 入门者的 AMD GPU 工程教程，硬件基线为 AI MAX 395 (gfx1151) + ROCm 7.12.0。教程主线是「硬件理解 → Profiling → 算子优化 → 推理优化 → 编译器 → Agent 自动化」。所有章节目前处于 Alpha 骨架阶段，正文待补。

---

## 目录

1. [tile-optimizer](#1-tile-optimizer)
2. [tritonBLAS](#2-tritonblas)
3. [bare-metal-1100 / t0-gpu](#3-bare-metal-1100--t0-gpu)
4. [triton-dejavu](#4-triton-dejavu)
5. [compiler-agent / cuasmrl](#5-compiler-agent--cuasmrl)
6. [gpu-sharing / HAMi](#6-gpu-sharing--hami)

---

## 1. tile-optimizer

**仓库**: `/mnt/luyuzhou/hpc/tile-optimizer/`
**作者**: 刘宇舟
**项目定位**: GEMM tile 尺寸优化与 Micro-to-E2E Gap 研究

### 可贡献章节

#### 第 16 章「Matmul 入门优化」— tile 选择的系统化方法

教程当前大纲只说了「观察 tile size 对性能的影响」，但没有给出选择依据。你的 `tile_agent/rdna3.py` 中的 `prune_search_space()` 函数定义了 6 条硬件约束规则，可以直接转化为教程正文：

**贡献方式**: 写一个「tile 大小怎么选」小节，包含：

1. **硬件约束规则表**（从 `rdna3.py:76-120` 提取）：
   - BLOCK_M ≤ M × 2（避免过度 padding）
   - BLOCK_K 是 WMMA_K (16) 的倍数
   - LDS 使用 < 32KB（双缓冲预算：`2 * (BM + BN) * BK * dtype_bytes`）
   - wave 利用率 ≥ 4 waves/CU
   - 寄存器估算 < 256 VGPRs/wave
   - BLOCK_N × dtype_bytes % 16 == 0（bank 友好）

2. **MoE 小矩阵的 tile 行为**（从 `experiments/rdna3/moe_tile_sweep.py` 的结果）：MoE 16×8192×2048 的最优 tile (16×64×32) vs 默认 tile (64×64×32) 差 31%，说明默认启发式在小 M 场景失效。

3. **一个决策流程图**：给定 (M, N, K)，先用约束规则裁剪，再按 wave 利用率和 M 利用率排序。

**对应代码目录**: `code/part3-hip-kernels/chapter16/`

#### 第 21 章「Triton 自动调参」— autotune 的局限性

你的 `PAPER_STORY.md` 中的 Micro-to-E2E Gap 框架是第 21 章「autotune 不是万能的」的核心素材。

**贡献方式**: 写一个「autotune 的边界」小节，包含：

1. **三个传导条件**（从 `PAPER_STORY.md` 提取）：
   - 后端 tile 可控（Triton 能指定 BLOCK_M/N/K）
   - 优化的 kernel 在热路径上（不是被调度框架覆盖的冷路径）
   - 实际 workload shape 分布与测试 shape 匹配

2. **反面案例**：Marlin bsm 在 M=126 上有 2.1× 加速，但端到端反而慢 4.3%（99% 的调用在 M=4，启发式已经最优）。

3. **tile 形状 > tile 面积**：128×64 vs 64×128 面积相同但性能差 1.42×（Blackwell 实测），说明 BLOCK_M=128 对 M=128 有完美的寄存器复用。

**对应代码目录**: `code/part4-triton/chapter21/`

#### 第 10 章「Omniperf 与硬件计数器进阶」— ISA diff 方法论

你的 `analysis/isa_diff.py` 实现了一种有价值的分析方法：对比同一 kernel 在不同 tile 配置下的 ISA 差异。

**贡献方式**: 在第 10 章的「进阶报告模板」中加一个「ISA 级对比分析」案例：

1. 用 `llvm-objdump --mcpu=gfx1100` 反汇编两个不同 tile 配置的 kernel
2. 统计 WMMA 指令前后的 `v_perm`/`v_pack`/`v_mov` 重排操作数
3. 对比 scratch spill 数和 stall 计数
4. 得出「哪个 tile 配置在 ISA 级更优」的结论

#### 第 7 章「性能优化的基本方法论」— Arithmetic Intensity 分区

你的 `tile_agent/blackwell.py` 中的 arithmetic intensity 分区思想（I < 10 = memory-bound，I ≥ 10 = compute-bound）可以放进第 7 章的「Memory-bound 与 Compute-bound」小节。

**贡献方式**: 写一个「怎么判断算子是访存瓶颈还是计算瓶颈」的量化方法，给出 arithmetic intensity 公式和分区阈值。

---

## 2. tritonBLAS

**仓库**: `/mnt/luyuzhou/hpc/tritonBLAS/`
**作者**: AMD ROCm 团队
**项目定位**: 基于 Origami 解析模型的无 autotune GEMM 库

### 可贡献章节

#### 第 16 章「Matmul 入门优化」— LDS 用量精确公式

你的 `include/tritonblas/origami.py` 中的 `estimate_triton_lds_bytes()` 给出了 Triton AMD 后端 LDS 用量的精确公式，在 gfx942 上 35/35 配置全部验证通过。

**贡献方式**: 在第 16 章的「LDS 缓存」小节加入：

```python
# Triton AMD 后端 LDS 用量估算
# ns == 1: max(A_bytes, B_bytes)          — 无流水线
# ns >= 2: (ns - 1) * (A_bytes + B_bytes) — 软件流水线
#
# 其中:
#   A_bytes = BLOCK_M * BLOCK_K * dtype_bytes
#   B_bytes = BLOCK_K * BLOCK_N * dtype_bytes
#   num_stages = 2 (Triton matmul 默认)
#
# gfx942 验证: 35/35 配置精确匹配 (Triton 3.6.0+rocm7.2.0)
```

这个公式回答了「tile 大了 LDS 够不够用」的问题，是 tile 选择的关键约束。

#### 第 16 章 — 调度策略概览

你实现了六种 GEMM kernel 变体。教程不需要教读者写这些 kernel，但可以在「简化版高性能 GEMM」之后加一个「库级 GEMM 的调度策略」概览框。

**贡献方式**: 写一页纸的概览，包含：

1. **Persistent GEMM**: 每个 workgroup 存活处理多个 tile，减少 kernel launch 开销
2. **Stream-K**: K 维度细粒度划分，当 tile 数 < CU 数时多个 workgroup 协作处理部分 tile，用锁协议聚合部分结果
3. **Work-Stealing**: 原子计数器动态分配下一个 tile，避免静态划分的负载不均

每个策略用一句话说明「解决什么问题」，配一张示意图。

#### 第 21 章「Triton 自动调参」— 解析模型 vs 暴力搜索

你的 `OrigamiMatmulSelector` 代表了 autotune 的另一种范式。

**贡献方式**: 在第 21 章的「选择最优 kernel config」小节加一个对比：

| 维度 | `@triton.autotune` | Origami 解析模型 |
|------|--------------------|-----------------| 
| 搜索方式 | 暴力枚举 + 实测 | 解析公式直接计算 |
| 首次开销 | 数秒到数分钟 | 微秒级 |
| 可解释性 | 黑盒（只知道哪个快） | 白盒（每个决策有公式依据） |
| 泛化能力 | 只覆盖测试过的 shape | 公式可泛化到任意 shape |
| 局限性 | 大搜索空间不可行 | 需要精确的硬件模型 |

#### 第 3 章「AMD GPU 体系结构」— Chiplet-aware scheduling

你的 `kernels/stages/indexing/pid_transforms.py` 中的 `chiplet_transform_chunked()` 是一个关键的多 chiplet 调度优化。

**贡献方式**: 在第 3 章的「CU/SIMD/Wavefront」小节中加一个「多 chiplet 的 L2 局部性」提示框，说明：
- MI300X 有 8 个 XCD，每个 XCD 有自己的 L2 cache
- 线性的 workgroup ID 分配会导致跨 XCD 的 L2 访问
- `chiplet_transform_chunked()` 通过重映射 PID 让每个 XCD 处理 L2 本地 tile

对 gfx1151 基线来说 chiplet 不直接适用，但概念值得提及。

---

## 3. bare-metal-1100 / t0-gpu

**仓库**: `/mnt/luyuzhou/hpc/bare-metal-1100/t0-gpu/`
**作者**: 刘宇舟
**项目定位**: 从零构建的 Rust GPU 编译器 + 裸机运行时 + 推理引擎

### 可贡献章节

#### 第 5 章「ROCm 软件栈与工具链」— HIP 背后的 KFD 原理

你的 `src/kfd/mod.rs` (3143 行) 是一份活的 KFD 教材。教程不需要教读者写 KFD 代码，但可以用它画一张「HIP → KFD → GPU」的分层图。

**贡献方式**: 写一个「HIP runtime 底层发生了什么」的补充说明：

1. **AQL dispatch packet**：64 字节的硬件格式，`header` 字段最后写入（原子性保证）
2. **dispatch 延迟对比**：bare-metal async 2.26μs vs HIP 2.6μs — 说明 HIP 的 overhead 在哪（ring buffer 空间检查、SYSTEM-scoped fence 的 L2 writeback）
3. **PM4-in-AQL hybrid**：VENDOR_SPECIFIC packet 嵌入 INDIRECT_BUFFER 做寄存器设置，再跟一个原生 KERNEL_DISPATCH

#### 第 16 章「Matmul 入门优化」— 超越教学级的 GEMM

你的 GEMM 实现在 RX 7900 XTX 上超越了 rocBLAS（2048³ 矩阵 117%）。

**贡献方式**: 在第 16 章的「与 rocBLAS 对比」小节加一个「差距从哪来」分析框：

| 优化手法 | 教学版 GEMM | 库级 GEMM (t0-gpu) |
|----------|------------|-------------------|
| WMMA fragment 布局 | 未明确 | 8-VGPR 对齐，lane 布局文档化 |
| LDS 双缓冲 | 单缓冲 | 双缓冲与 K-loop 流水线配合 |
| 累加器交错 (ILP) | 无 | 交错不同 acc group 隐藏 WMMA 延迟 |
| Split-K | 无 | 大 K 维度拆分到多 block 并行 |
| WGP 模式 | 无 | 2-CU workgroup 的调度策略 |

#### 第 10 章「Omniperf 与硬件计数器进阶」— ISA 级 profiling 方法

你的 `benchmarks/profile_gemm.py` 实现了一套完整的 ISA 深度分析流程。

**贡献方式**: 在第 10 章的「进阶报告模板」中提供一个 ISA 分析报告范例，包含：
- VGPR/SGPR 数量、LDS 大小、WGP mode（从 kernel descriptor 提取）
- WMMA 调度模式：累加器交错、双链重叠
- waitcnt 值分布和 barrier 频率
- 指令混合比例：VALU/VMEM/LDS/SALU/CTRL 占比
- 软件流水线深度

#### 第 11-16 章 — GPU hang 调试案例

你的文档中有大量 GPU hang 的 root cause 分析。

**贡献方式**: 在第 11 章「错误检查与调试」中加一个「常见 GPU hang 模式」附录：
- VCC 残留导致 wave 卡死
- EXEC 掩码不平衡
- 缺少 waitcnt 导致数据冒险
- LICM 移动了 barrier 之前的指令
- cooperative load 的 race condition

每个模式给出：症状、ISA 级原因、如何避免。

---

## 4. triton-dejavu

**仓库**: `/mnt/luyuzhou/hpc/triton-dejavu/`
**作者**: Burkhard Ringlein (IBM Research)
**项目定位**: Triton autotuner 缓存持久化框架

### 可贡献章节

#### 第 21 章「Triton 自动调参」— autotune 的工程问题

教程当前只考虑了「怎么找到最优 config」，但没有考虑「找到之后怎么办」。

**贡献方式**: 写一个「autotune 的工程挑战」小节：

1. **缓存持久化**：autotune 结果按 `GPU型号 + Triton版本 + ROCm版本 + kernel源码哈希 + config哈希` 索引到磁盘 JSON。下次启动直接恢复，零开销。
2. **JIT 启动开销**：Triton 的完整 dispatch 路径（类型检查、特化检测、绑定）每次调用数百微秒。JitCache 缓存编译后的 kernel handle，降到 20-50μs。
3. **Cache miss 回退**：新 shape 没有缓存怎么办？三种策略：完整 benchmark（慢）、贝叶斯优化（中）、启发式回退（快但不精确）。
4. **Crash 隔离**：autotune 过程中某个 config 导致 kernel crash（非法内存访问），隔离进程模式保证不影响其他 config 的测试。

#### 第 7 章「性能优化的基本方法论」— 可信 benchmark 方法

你的 `testing.py` 中的 benchmark 实现比教程当前规划的方法论更具体。

**贡献方式**: 在第 7 章的「如何设计一个可信的 benchmark」小节补充：

1. **L2 cache flushing**：维护 256MB buffer，每次计时前清零，防止 L2 cache 命中扭曲结果
2. **CUDA graph capture**：捕获 N 次 kernel 调用到 graph 中重放，最小化 host-side 开销
3. **分位数报告**：不只报 median，还要报 20th/80th 分位数，暴露尾部延迟
4. **隔离进程**：benchmark 在子进程中运行，kernel crash 不影响主进程

#### 第 20 章「Triton Attention 优化」— Flash Attention 参考实现

你的 `tests/rocm_flash_attention.py` 是一个完整的 Flash Attention v2 Triton 实现。

**贡献方式**: 作为第 20 章的参考代码，重点标注以下技巧：

1. **`exp2` 替代 `exp`**：硬件原生支持 `2^x` 但不支持 `e^x`，用 `exp2(x * 1.44269504089)` 替代 `exp(x)` 性能更好
2. **Block pointer API**：`tl.make_block_ptr` + `tl.advance` 替代原始指针算术，自带边界检查
3. **Causal masking 优化**：计算 `n_blocks` 和 `masked_blocks` 跳过全 mask 区域，避免无用计算

---

## 5. compiler-agent / cuasmrl

**仓库**: `/mnt/luyuzhou/hpc/compiler-agent/cuasmrl/`
**作者**: 刘宇舟
**项目定位**: RL 驱动的 SASS 指令重排序 + Tile Size Reasoning Agent

### 可贡献章节

#### 第 27 章「AI 编译器到底在优化什么」— ISA 级优化的可能性

cuasmrl 在 Triton 编译管线的最后一层（SASS）做文章。这个案例展示了优化可以发生在编译管线的每一层。

**贡献方式**: 在第 27 章的「从 kernel 到 ISA」小节加一个「编译管线各层的优化机会」图：

```
Triton DSL    →  tile 大小、循环顺序、内存访问模式
    ↓
MLIR / SSA    →  算子融合、常量折叠、死代码消除
    ↓
PTX           →  指令选择、寄存器分配
    ↓
SASS          →  指令调度、内存指令重排 (cuasmrl 在这一层)
    ↓
cubin         →  最终二进制
```

说明每一层的优化粒度和收益上限不同：高层优化（如 tile 选择）影响大但粗粒度，低层优化（如 SASS 重排）影响小但细粒度。

#### 第 6 章「第一个 AMD GPU 程序」— 指令调度对性能的影响

cuasmrl 的 `sample.py` 中的 action masking 逻辑展示了指令调度的约束：RAW/WAR 依赖、scoreboard 冲突、stall count 窗口。

**贡献方式**: 在第 6 章或第 11 章中加一个「为什么指令顺序 matters」的简短说明：
- GPU 的 wavefront 是乱序执行的，但内存指令的顺序影响 stall
- 一条 LDG（全局内存加载）延迟 ~300 cycles，如果后续指令依赖它的结果，wave 会 stall
- 把不依赖 LDG 结果的指令插在 LDG 和它的使用者之间，可以隐藏延迟
- 这就是 cuasmrl 做的事情：在保持正确性的前提下重排内存指令

---

## 6. gpu-sharing / HAMi

**仓库**: `/mnt/luyuzhou/hpc/gpu-sharing/`
**作者**: 刘宇舟
**项目定位**: Kubernetes GPU 虚拟化与多租户调度

### 与 hello-gpu 的关系

gpu-sharing 属于 hello-ai-infra-platform（三本书的第三本）的内容范畴，与 hello-gpu（单卡视角）没有直接章节对应。

### 间接贡献

#### vLLM TTFT 测量方法（可选）

HAMi 的 `benchmarks/ai-benchmark/benchmark.py` 实现了一套 streaming LLM 推理的 TTFT + per-token latency 测量方法。如果你在写第 25 章「LLM 单卡推理性能分析入门」时需要一个 TTFT 测量的参考实现，这个代码可以作为补充。

**贡献方式**: 在第 25 章的「TTFT、TPOT 与单卡吞吐」小节中，如果需要一个端到端的测量脚本范例，可以从 HAMi benchmark 中提取简化版。

---

## 贡献流程

1. **Fork 并 Clone**: `git clone https://github.com/datawhalechina/hello-gpu`
2. **选择章节**: 根据上表找到你要贡献的章节目录（`docs/partN-name/chapterN/`）
3. **编辑正文**: 修改 `index.md`，在对应的小节中补充内容。注意保持已有的 frontmatter 和章节编号格式
4. **补充代码**: 在对应的 `code/partN-name/chapterN/` 目录中放入可运行的实验代码
5. **提 PR**: 向 `main` 分支提 Pull Request

### 内容规范

- 语言：中文，技术术语可保留英文
- 代码注释：英文
- 性能数字：必须标注硬件型号、ROCm 版本、Triton 版本
- 图片：放在 `docs/partN-name/chapterN/images/` 目录，使用 PNG 格式（构建时自动转 WebP）
- 交叉引用：使用 `::: figure fig-id` 容器和 `@fig-id` 语法
- 公式：使用 LaTeX 语法（MathJax3 渲染）

### 联系方式

- Issue: https://github.com/datawhalechina/hello-gpu/issues
- 项目负责人: 刘伟鸿 (DataWhale)
