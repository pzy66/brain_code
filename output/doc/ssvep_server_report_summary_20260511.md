# SSVEP 服务器报告汇总

更新：2026-05-11 19:15（balanced 全量对照已完成）

## 当前结论
- 首选部署候选：LRT `win2_me2_sm3_lrtmw`，idle_fp=0.837，recall=0.885，recall@2.5s=0.778，latency=2.297s。
- balanced 也通过：`win2_me2`，idle_fp=0.847，recall=0.882，recall@2.5s=0.765，latency=2.331s。
- 下一步：固化 LRT candidate，并针对 NS2 false positive 调门控。