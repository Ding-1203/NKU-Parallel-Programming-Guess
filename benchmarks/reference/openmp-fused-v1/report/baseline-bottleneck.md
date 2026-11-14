# openmp-fused-v1 result

本报告评估 BatchPlan 驱动的候选生成–AVX2 哈希融合；headline 与 correctness/profiling 严格分离。

## 公平性与稳定性

- 工作负载：10,000,000 candidates，batch=1,000,000，CPU affinity=0,2,4,6,8,10,12,14。
- 正式运行：2 次预热 + 7 次测量。
- core_total 中位数：0.120112 s；bootstrap 95% CI [0.115608, 0.125529] s。
- CV：4.05%（阈值 5%）；状态：稳定，可作为 speedup 基准。
- 吞吐中位数：83.256 M candidates/s。离群点被标记但未删除。

## 正确性门禁

exact 字段 generated/hash_guard/sig1/sig2/candidate_hits/cracked_unique 全部匹配；7 次性能运行的 generated/hash_guard 也一致。未来 relaxed 版本必须独立展示质量变化，不进入 strict speedup 排名。

## 强扩展与累计加速

- 1T: 0.525921 s, 1.00×, efficiency 100.0%
- 2T: 0.270691 s, 1.94×, efficiency 97.1%
- 4T: 0.181551 s, 2.90×, efficiency 72.4%
- 8T: 0.120112 s, 4.38×, efficiency 54.7%
- 16T: 0.136628 s, 3.85×, efficiency 24.1%
- 8T vs openmp-md5-v1: 1.95×, 95% CI [1.87, 2.04]
- 8T vs serial-md5-avx2-v1: 4.47×, 95% CI [4.19, 4.68]
- 8T vs serial-md5-fast-v1: 11.07×, 95% CI [10.57, 11.53]
- 8T vs serial-heap-v1: 17.49×, 95% CI [16.72, 18.29]
- 8T vs baseline-v1: 39.67×, 95% CI [37.71, 41.41]

## 瓶颈结论

1. sampler_init 0.008930 s，占 detailed core 6.2%。
2. 确定性 heap（串行）：运行期 pop/expand/insert 中位数 0.000263 s；线性搬移元素 0。
3. batch plan 0.000128 s；fused generate + AVX2 worker 0.102956 s；ordered guard merge 0.013328 s，占 detailed core 9.2%。
4. 模型加载：headline load_model 中位数 1.136696 s。它影响 end_to_end，但不进入 core_total。gprof 最大热点 `pcfg::Sha256::transform(unsigned char const*)`（40.9%）包含模型加载与完整性校验；该采样比例不能解释为 core_total 的墙钟占比。
5. 总 RSS 中位数 fused 664.0 MiB、OpenMP-MD5 664.0 MiB；模型主导峰值，因此结构性验收以正式主循环不再持有完整 vector<string> 为准。

## 下一项优化目标

融合节点内核已完成；下一阶段可评估 MPI RangeTask 分发或进一步降低有序 guard 串行尾部。 任何优化必须复用本报告的模型、hash、CPU affinity、2+7 协议和正确性门禁。
