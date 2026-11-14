# baseline-bottleneck

本报告只定位瓶颈，不改变采样算法。headline 数据来自 instrumentation off；阶段拆分与 gprof 仅用于解释，不能进入加速比。

## 公平性与稳定性

- 工作负载：10,000,000 candidates，batch=1,000,000，CPU affinity=0。
- 正式运行：2 次预热 + 7 次测量。
- core_total 中位数：4.764346 s；bootstrap 95% CI [4.731815, 4.814593] s。
- CV：1.48%（阈值 3%）；状态：稳定，可作为 speedup 基准。
- 吞吐中位数：2.099 M candidates/s。离群点被标记但未删除。

## 正确性门禁

exact 字段 generated/hash_guard/sig1/sig2/candidate_hits/cracked_unique 全部匹配；7 次性能运行的 generated/hash_guard 也一致。未来 relaxed 版本必须独立展示质量变化，不进入 strict speedup 排名。

## 首轮瓶颈结论

1. sampler_init：10M detailed 中位数 2.565549 s，占 core_total 54.9%。规模扫描显示这一固定成本在 candidate limit 增大时被摊薄。
2. 有序 vector：运行期 pop/expand/insert 中位数 0.090851 s；估算搬移元素 1,187,587,194。gprof 最大热点为 `void std::vector<pcfg::SearchState, std::allocator<pcfg::SearchState> >::_M_insert_aux<pcfg::SearchState>(__gnu_cxx::__normal_iterator<pcfg::SearchState*, std::vector<pcfg::SearchState, std::allocator<pcfg::SearchState> > >, pcfg::SearchState&&)`，占 44.4%。
3. scalar MD5：10M detailed 的 hash 阶段 1.905980 s，约 190.6 ns/candidate，是主要随候选数增长的热点。
4. 模型加载：headline load_model 中位数 1.164759 s。它影响 end_to_end，但不进入 core_total，因此训练和加载不能混入采样加速比。

## 下一项优化目标

优先替换 sampler 初始化及运行期有序 vector 的搬移机制，同时保持 exact 候选顺序不变；之后再评估 scalar MD5。任何优化必须复用本报告的模型、hash、CPU affinity、2+7 协议和正确性门禁。
