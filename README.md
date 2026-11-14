# NKU PCFG Serial Baseline

这是期末并行程序设计实验的全新串行基线。离线训练与在线采样完全解耦；后续 MPI、OpenMP、SIMD 或数据结构优化都必须复用这里的模型身份、候选语义、正确性门禁和测评协议。

## Baseline 边界

- C++17、CMake、严格串行 PCFG。
- 严格概率优先的有序 `vector<SearchState>` 队列。
- 普通字符串批处理与 scalar MD5。
- 不链接 MPI/OpenMP，不包含 SIMD intrinsic。
- 新模型格式与新结果 schema，不兼容阶段作业。
- `baseline-v1.0` 只定位瓶颈，不包含采样算法优化。

## 固定构建

可比性能版本只能使用 `benchmark` preset：

```bash
cmake --preset benchmark
cmake --build --preset benchmark
ctest --preset benchmark
```

该 preset 固定 `-O2 -DNDEBUG -march=native` 并导出完整编译命令。独立 `gprof` preset 带 `-pg`，其结果只能用于热点解释，不能用于 speedup。

## 数据与模型

真实数据和模型不提交 Git。默认正式身份：

- `data/raw/rockyou.txt`：完整 32,575,499 行 RockYou。
- `artifacts/models/rockyou-full.pcfg`：全量确定性模型。
- 源数据、模型、构建产物、普通实验和 `.vscode/` 均被忽略。

准备数据：

```bash
./scripts/prepare_data.sh
```

若模型不存在，使用离线训练命令生成：

```bash
build/benchmark/pcfg train \
  --input data/raw/rockyou.txt \
  --output artifacts/models/rockyou-full.pcfg \
  --train-limit 0
```

## 采样与 instrumentation

headline 性能必须使用 `--instrumentation off`。该路径不执行高频阶段计时，只保留 `core_total`、`end_to_end` 和正确性必需计算。

```bash
build/benchmark/pcfg sample \
  --model artifacts/models/rockyou-full.pcfg \
  --limit 10000000 \
  --batch-size 1000000 \
  --threads 1 \
  --instrumentation off
```

瓶颈定位使用 `detailed`，可每 100,000 candidates 写一条 JSONL trace：

```bash
build/benchmark/pcfg sample \
  --model artifacts/models/rockyou-full.pcfg \
  --limit 1000000 \
  --batch-size 1000000 \
  --instrumentation detailed \
  --trace-out results/trace.jsonl
```

结果 schema 仍为 `pcfg-baseline-result-v2`。`--threads N` 默认 1；多线程只并行 batch 内的 AVX2 MD5 group，并按原候选顺序归并 `hash_guard`。detailed 模式额外给出 `hash_compute` 与 `hash_merge`，用于区分并行 MD5 和顺序归并。

## 统一实验入口

```bash
./scripts/bench run     --output results/baseline-v1
./scripts/bench profile --output results/baseline-v1
./scripts/bench analyze --input  results/baseline-v1
```

当前 OpenMP-MD5 实验使用 `--config benchmarks/configs/openmp-md5-v1.json`；融合实验使用 `--config benchmarks/configs/openmp-fused-v1.json`；runner 固定 1/2/4/8/16T 的 CPU places，其中 8T 是正式 headline。该亚秒级 WSL 多线程点使用 5% CV 验收阈值；原有 baseline-v1.0 的 3% 契约不变。

`run` 执行身份预检、构建、CTest、exact 正确性门禁、2 次预热和 7 次正式 10M 测量。正式配置在工作树 dirty 或数据/模型摘要不匹配时直接拒绝。

`profile` 执行 100K/1M/10M 规模扫描、64K/256K/1M batch 敏感度、10M gprof、可选 strace，并明确记录 perf 是否可用。`analyze` 生成 CSV/JSON、bootstrap 95% CI、PDF 图和轻量 HTML。

完整协议、统计口径、预计耗时和 canonical 包说明见 [docs/BENCHMARKING.md](docs/BENCHMARKING.md)。

## 快速 smoke

```bash
./scripts/bench run --config benchmarks/configs/smoke.json --output results/smoke
./scripts/bench analyze --input results/smoke
```

smoke 使用独立合成数据，只验证工具链，不可用于性能结论。

## Canonical reference

正式实验分析通过后：

```bash
./scripts/bench freeze \
  --input results/baseline-v1 \
  --destination benchmarks/reference/baseline-v1
```

精简包只提交身份 manifest、7 次 performance JSON、correctness、summary、profiler 表、关键图与报告。RockYou、模型、完整日志、gmon 和 trace 继续留在忽略的 `results/`。

## MPI + OpenMP + AVX2

MPI 前端是可选目标，原 `pcfg` 保持不链接 MPI。Ubuntu/WSL 依赖：

```bash
sudo apt-get install -y openmpi-bin libopenmpi-dev
cmake --preset mpi-benchmark
cmake --build --preset mpi-benchmark
ctest --preset mpi-benchmark
```

`pcfg_mpi` 仅支持采样。Rank 0 维护确定性 Heap 并广播紧凑 `BatchPlan`；
所有 Rank 计算连续 slice，使用 `MPI_Gatherv` 按序收集 folded digest，最后由
Rank 0 执行原 `hash_guard`。候选字符串不会在 performance 路径经过 MPI。

正式实验入口：

```bash
./scripts/bench-mpi run     --output results/mpi-fused-exact-v1
./scripts/bench-mpi profile --output results/mpi-fused-exact-v1
./scripts/bench-mpi analyze --input  results/mpi-fused-exact-v1
```

runner 在每个大阶段前打印完整命令、Rank/线程/CPU 映射和 ETA。10M 用于六项
exact 历史门禁；100M 是 MPI 持续吞吐 headline。正式 tag 要求 100M CV 不超过
5%，且相对 `openmp-fused-v1.0` 的 bootstrap 95% speedup CI 下界大于 1。
图表只生成 PDF，HTML 不使用在线依赖。
