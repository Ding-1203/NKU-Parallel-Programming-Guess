# serial-md5-avx2-v1 result

本报告评估 AVX2 八路跨候选 MD5；headline 与 profiling 数据严格分离。

## 公平性与稳定性

- 工作负载：10,000,000 candidates，batch=1,000,000，CPU affinity=0。
- 正式运行：2 次预热 + 7 次测量。
- core_total 中位数：0.537394 s；bootstrap 95% CI [0.525447, 0.541239] s。
- CV：1.93%（阈值 3%）；状态：稳定，可作为 speedup 基准。
- 吞吐中位数：18.608 M candidates/s。离群点被标记但未删除。

## 正确性门禁

exact 字段 generated/hash_guard/sig1/sig2/candidate_hits/cracked_unique 全部匹配；7 次性能运行的 generated/hash_guard 也一致。未来 relaxed 版本必须独立展示质量变化，不进入 strict speedup 排名。

## 瓶颈结论

1. sampler_init：10M detailed 中位数 0.009027 s，占 core_total 1.7%。规模扫描显示这一固定成本在 candidate limit 增大时被摊薄。
2. 确定性 heap：运行期 pop/expand/insert 中位数 0.000185 s；线性搬移元素 0。gprof 最大热点为 `pcfg::Sha256::transform(unsigned char const*)`，占 40.7%。
3. AVX2 MD5：10M detailed 的 hash 阶段 0.424164 s，约 42.4 ns/candidate，是主要随候选数增长的热点。
4. 模型加载：headline load_model 中位数 1.057565 s。它影响 end_to_end，但不进入 core_total，因此训练和加载不能混入采样加速比。

## 下一项优化目标

AVX2 阶段已完成；下一优化目标由新的 profiling 结果决定。 任何优化必须复用本报告的模型、hash、CPU affinity、2+7 协议和正确性门禁。
