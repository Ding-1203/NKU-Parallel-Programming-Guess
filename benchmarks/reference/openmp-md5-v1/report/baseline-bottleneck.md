# openmp-md5-v1 result

本报告只评估 batch 内 AVX2 MD5 group 的 OpenMP 并行；候选生成、heap 与有序 hash_guard 归并保持串行。

## 公平性与稳定性

- 工作负载：10,000,000 candidates，batch=1,000,000，CPU affinity=0,2,4,6,8,10,12,14。
- 正式运行：2 次预热 + 7 次测量。
- core_total 中位数：0.234773 s；bootstrap 95% CI [0.229879, 0.236878] s。
- CV：1.65%（阈值 5%）；状态：稳定，可作为 speedup 基准。
- 吞吐中位数：42.594 M candidates/s。离群点被标记但未删除。

## 正确性门禁

exact 字段 generated/hash_guard/sig1/sig2/candidate_hits/cracked_unique 全部匹配；7 次性能运行的 generated/hash_guard 也一致。未来 relaxed 版本必须独立展示质量变化，不进入 strict speedup 排名。

## 强扩展与累计加速

- 1T: 0.550618 s, 1.00×, efficiency 100.0%
- 2T: 0.365492 s, 1.51×, efficiency 75.3%
- 4T: 0.265016 s, 2.08×, efficiency 51.9%
- 8T: 0.234773 s, 2.35×, efficiency 29.3%
- 16T: 0.263243 s, 2.09×, efficiency 13.1%
- 8T vs baseline-v1: 20.29×, 95% CI [19.98, 20.83]
- 8T vs serial-heap-v1: 8.95×, 95% CI [8.81, 9.22]
- 8T vs serial-md5-fast-v1: 5.67×, 95% CI [5.59, 5.80]
- 8T vs serial-md5-avx2-v1: 2.29×, 95% CI [2.22, 2.35]

## 瓶颈结论

1. sampler_init 0.008683 s，占 detailed core 3.1%。
2. 确定性 heap（串行）：运行期 pop/expand/insert 中位数 0.000280 s；线性搬移元素 0。
3. 串行 candidate generation 0.151072 s，已成为最大 core 阶段；AVX2 compute 0.089938 s，ordered guard merge 0.011333 s。
4. 模型加载：headline load_model 中位数 1.130232 s。它影响 end_to_end，但不进入 core_total。gprof 最大热点 `pcfg::Sha256::transform(unsigned char const*)`（39.4%）包含模型加载与完整性校验；该采样比例不能解释为 core_total 的墙钟占比。

## 下一项优化目标

依据 1T/8T 阶段拆分评估生成–哈希流水化；本轮不修改采样算法。 任何优化必须复用本报告的模型、hash、CPU affinity、2+7 协议和正确性门禁。
