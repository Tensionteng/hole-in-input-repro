# S5 晨间汇报骨架（过夜验证实验汇总）

> 用途：明早汇总时在此骨架上填充 Track A/B/C 结果。Part 0 为已抽查验证的 pilot 结论（数值已与 s5_missing_results.json 逐项核对），不会变动。

## Part 0 — Pilot 已确认结论（s5_missing_results.json，3 模型 × 3 数据集，已验证）

设定：context 512 / horizon 96，ETTh1+ETTm1+weather，测试段为时间轴末 20%，relMSE = 缺失配置 MSE / 无缺失基线 MSE。模型：chronos-bolt-base（patch=16，mean-abs scaling）、chronos-t5-small（量化分词）、timesfm-2.5-200m（patch=32）。

1. **MCAR + 线性插值 ≈ 免疫**：p=0.5 时 relMSE 1.04–1.26；p=0.7 时 1.07–1.62。weather 甚至变好（bolt 0.88，t5 0.77；插值去噪效应）。
2. **zero-fill 是灾难，通道因架构而异**：p=0.5 平均 relMSE 5.87（weather 13.65），p=0.7 平均 13.74（weather 34.07）。oracle-scale 探针：Bolt 在 p=0.5 恢复 79–83% 损伤（内部归一化统计量污染主导）；t5 全部位无效（值污染/token 分布失真主导）；TimesFM 部分恢复（6.07→3.00 @p=0.5）。
3. **block(24) > MCAR（同率），尽管损坏 patch 更少**：bolt ETTh1 p=0.7：block 1.51 vs mcar 1.26（cpf 0.69 vs 1.00）。t5（patch=1）是最干净对照：同 cpf 下 block 仍更差 → 损伤由填充的插值误差介导，与 patch 污染无关。
4. **MNAR（值审查）≈ 2× MCAR，且性质为系统性偏差**：p=0.7 mnar_high relMSE 2.0–2.7（ETT）；top-decile 分析（bolt ETTh1 p=0.7）：极端 future 点 MSE 56.2 vs 普通点 23.7（比值 2.4；mcar 下仅 0.7）→ 模型系统性低估峰值。oracle-scale 在 MNAR 下全面失效（"恢复率" -3% ~ -225%）→ MNAR 损伤 = 信息丢失+偏差，不可由标度修复。

机制层级：值污染/偏差（MNAR、block）≫ 归一化统计量污染（mcar zero-fill，架构相关）≫ patch token 污染（已否决）。

## Part A — 修复方法（占位：s5_fix_results.json / s5_fix.png / s5_fix_notes.md）

- Fix 0 原生 NaN 行为：【待填】
- Fix 1 observed-stats 重标定：各机制恢复率【待填】
- Fix 2 Bolt mask 感知输入层手术：是否成功 / 回退原因【待填】；block/mnar 网格恢复率【待填】

## Part B — 真实数据验证（占位：s5_real_results.json / s5_real.png / s5_real_notes.md）

- 数据集与天然缺失率统计：【待填；PhysioNet 或 USHCN，若失败记录原因】
- 真实缺失下填充策略排序 / zero-fill 灾难是否复现：【待填】
- 重标定在真实（信息性）缺失下的效果：【待填】
- 叠加人工缺失压力测试：【待填】

## Part C — 补充机制（占位：s5_extra_results.json / s5_extra.png / s5_extra_notes.md）

- block 长度扫描 → 插值失效临界尺度：【待填】
- MAR（时间聚集、值独立）对照 → 聚集 vs 值依赖的归因：【待填】
- 预测区间校准随缺失率的退化：【待填】
- Moirai 扩展面板（可选）：【待填 / 跳过原因】

## 论文故事线（晨间定稿用）

1. 恐惧被夸大：随机缺失在合理插值下近乎无害（至 70%）。
2. 真正的杀手有两张脸：(i) 预处理污染内部归一化统计量（可修，黑盒重标定）；(ii) 信息性/结构性缺失引入系统偏差（须 mask/机制感知方法）。
3. 修复方法与机制一一对应，实验验证恢复率。
4. 真实数据复现 + 校准/临界尺度等补充证据。
负面结果与局限：【晨间统一填写】
