# serial-heap-v1 result

本报告评估确定性 heap 队列优化；headline 与 profiling 数据严格分离。

## 公平性与稳定性

- 工作负载：10,000,000 candidates，batch=1,000,000，CPU affinity=0。
- 正式运行：2 次预热 + 7 次测量。
- core_total 中位数：2.100259 s；bootstrap 95% CI [2.081462, 2.133530] s。
- CV：1.83%（阈值 3%）；状态：稳定，可作为 speedup 基准。
- 吞吐中位数：4.761 M candidates/s。离群点被标记但未删除。

## 正确性门禁

exact 字段 generated/hash_guard/sig1/sig2/candidate_hits/cracked_unique 全部匹配；7 次性能运行的 generated/hash_guard 也一致。未来 relaxed 版本必须独立展示质量变化，不进入 strict speedup 排名。

## 瓶颈结论

1. sampler_init：10M detailed 中位数 0.011583 s，占 core_total 0.6%。规模扫描显示这一固定成本在 candidate limit 增大时被摊薄。
2. 确定性 heap：运行期 pop/expand/insert 中位数 0.000215 s；线性搬移元素 0。gprof 最大热点为 `pcfg::md5(std::basic_string_view<char, std::char_traits<char> >)`，占 55.6%。
3. scalar MD5：10M detailed 的 hash 阶段 1.980630 s，约 198.1 ns/candidate，是主要随候选数增长的热点。
4. 模型加载：headline load_model 中位数 1.154572 s。它影响 end_to_end，但不进入 core_total，因此训练和加载不能混入采样加速比。

## 下一项优化目标

heap 优化已完成；下一候选目标是 scalar MD5。 任何优化必须复用本报告的模型、hash、CPU affinity、2+7 协议和正确性门禁。
