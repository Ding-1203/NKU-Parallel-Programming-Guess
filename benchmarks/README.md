# Benchmark assets

- `configs/baseline.json`：baseline-v1.0 正式 10M/2+7 与 profiling 协议。
- `configs/smoke.json`：小型工具链验收，不用于性能结论。
- `reference/baseline-v1/`：正式运行后冻结的精简 canonical reference。

canonical 包可通过以下命令从原始实验重建：

```bash
./scripts/bench analyze --input results/baseline-v1
./scripts/bench freeze --input results/baseline-v1 --destination benchmarks/reference/baseline-v1
```

模型、RockYou、gmon、trace 与完整日志不会进入此目录。
