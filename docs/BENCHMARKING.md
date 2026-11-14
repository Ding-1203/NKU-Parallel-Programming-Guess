# baseline-v1.0 公平测评契约

## 1. Headline 协议

正式 speedup 的唯一 headline 指标是 `core_total`，吞吐率为 `generated/core_total`。模型加载单独进入 `end_to_end`；训练时间永远排除。

固定工作负载：

- 全量 RockYou 模型；
- 10,000,000 candidates；
- batch size 1,000,000；
- `--instrumentation off`；
- `taskset -c 0`；
- benchmark controller 自动移出 CPU 0 及其 SMT sibling，隔离前后 affinity 写入 manifest；
- 2 次预热，随后 7 次测量；
- benchmark preset：`-O2 -DNDEBUG -march=native`。

每个点报告中位数、IQR、MAD、CV、范围和固定种子的 bootstrap 95% CI。Tukey 离群点只标记、不删除。CV > 3% 时标记为 unstable，禁止作为最终 speedup。

预计本机耗时（Ryzen 7 8845H / WSL，随系统负载变化）：

| 操作 | 预计时间 |
|---|---:|
| smoke run + analyze | 20–40 秒 |
| headline run（正确性 + 2+7） | 2–3 分钟 |
| detailed 规模与 batch 扫描 | 3–5 分钟 |
| 10M gprof + strace | 20–50 秒 |
| 全套 run + profile + analyze | 6–9 分钟 |

执行者应在每次大实验启动前重新给出基于当前配置的时长预估。

## 2. 正确性与比较类别

`exact` 版本必须匹配：

- `generated`
- `hash_guard`
- `sig1`
- `sig2`
- `candidate_hits`
- `cracked_unique`

未来 `relaxed` 版本只验证预算与内部完整性；破解质量变化进入质量–吞吐图，不能进入 strict speedup 排名。分析器的 speedup 输入只接纳 `comparison_class=exact`。

## 3. Reference identity

`bench run` 固化：

- Git SHA、describe、dirty 状态；
- benchmark config SHA；
- 数据、模型、二进制 SHA-256；
- 模型预期记录数；
- CMake cache 与 `compile_commands.json`；
- 编译器、CMake、Python 与绘图库版本；
- CPU 拓扑、内存、WSL 内核、进程 affinity；
- `taskset`、`gprof`、`strace`、`perf`、`/usr/bin/time` 可用性。

正式配置默认 `require_clean=true`。dirty、数据 hash、模型 hash 或可选 binary hash 不匹配时拒绝运行。

## 4. Headline 与 profiling 分离

`off` 不在 pop/expand/generate/hash/verify 高频路径调用阶段计时器。`detailed` 才记录：

- sampler 初始化；
- queue pop；
- queue expand + ordered vector insert；
- candidate generation；
- scalar MD5；
- verification/signature；
- 初始/最大队列、pop/insert/expand/completed PT；
- pop/insert 估算搬移元素；
- batch/range 次数、候选字节和长度范围。

启用 `--trace-out` 后，每 100,000 candidates 写累计阶段和队列状态。trace 会增加观测开销，只用于瓶颈迁移图。

`gprof` preset 独立构建，结果不能与 benchmark preset 的 headline 时间混用。`perf` 不可用时必须记录为 unavailable/permission-denied，不伪造数据。

## 5. 实验目录

```text
results/<experiment>/
  manifest/   identity、构建和协议
  runs/       correctness、warm-up/measurement
  profiles/   detailed、gprof、strace、trace
  summary/    CSV 与聚合 JSON
  figures/    PDF
  report/     Markdown、self-contained HTML、visualization manifest
```

普通实验目录被 Git 忽略。只有 `benchmarks/reference/baseline-v1/` 的精简 canonical 包进入 Git。

## 6. 重建与未来比较

重建当前报告：

```bash
./scripts/bench analyze --input results/baseline-v1
```

比较未来 exact 版本：

```bash
./scripts/bench analyze \
  --input results/baseline-v1 \
  --compare results/optimized-exact-v1
```

每张图的源数据和生成命令记录在 `report/visualization-manifest.json`。HTML 不使用在线 CDN；PDF 与报告数值来自同一 summary/profile 数据。

## 7. MPI sustained-throughput 协议

`mpi-fused-exact-v1` 使用独立 `mpi-benchmark` preset 和 `pcfg_mpi`。
总计算资源固定为 8 个物理核，扫描 `1x8 / 2x4 / 4x2 / 8x1`。OpenMPI rankfile
配合 `--use-hwthread-cpus` 绑定偶数硬件线程；correctness 日志中的 Rank 绑定
必须完整且不能出现同核 `BB`。

- 10M：四种配置分别执行六项 exact 门禁；选中的 hybrid 另做 2+7 历史测量。
- 100M：四种配置 1+3 校准；4x2 只有至少快 3% 才取代 2x4。
- 选中 hybrid 与从 `openmp-fused-v1.0` 构建的直接 OpenMP 对照分别重新 2+7。
- `core_total` 包含 MPI、MD5、digest gather 和顺序 `hash_guard`；模型加载单列。
- detailed 只运行 1x8 与选中 hybrid，各记录每 1M batch 的 compute/sync/gather。
- MPI 多 Rank 不使用 gprof 解释墙钟关键路径。

常用命令：

```bash
./scripts/bench-mpi run --config benchmarks/configs/mpi-fused-exact-v1.json
./scripts/bench-mpi profile
./scripts/bench-mpi analyze
./scripts/bench-mpi freeze
```

每个大型阶段开始前显示完整命令和预计时长，首个 warm-up 后更新剩余 ETA。
普通结果保存在 `results/mpi-fused-exact-v1/`；canonical reference 排除模型、
RockYou、日志、time 文本、rankfile 和原始 batch trace。
