# S21 — GRU-D 式缺失感知输入化（informative missingness 搬到 bolt 输入层）

> 状态：**预注册冻结于 2026-08-14，冒烟（`s21_tiny.log`）通过之后、正式训练/评测运行之前**。
> 冒烟已证明：特征与独立 numpy 参考实现一致（max|diff| 3e-8）；值置换下特征逐位不变（防泄漏断言）；
> 零初始化投影与原生 encode 逐位一致（max|diff| = 0.0）；三变体 step-1 损失 = 37.6535 = S6 记录值；
> 掩码流与 S6 评测流一致。除"结果"一节外，设计/判定规则均在跑数前写下。

## 1. 思想与假设

GRU-D（Che et al. 2018）证明：对 RNN，mask + time-since-last-observation（+ 衰减）是有效的缺失
信号——**缺失模式本身携带信息**（informative missingness）。chronos-bolt 原生只有拼接进 patch
embedding 的 0/1 观测标志（S8 已证该通道有强因果作用，但只走 embedding/value 通路）。S20 证明
文本机制提示是惰性通道（退化为常数位移）——机制信息必须走**结构化输入**。S21 把 GRU-D 思想搬到
TSFM 输入层：在 bolt 的 patch embedding 上追加由观测指示变量计算的结构化缺失特征，问：

> **更丰富的结构化缺失特征，在 SFT-mb 已就位的情况下，能否超过 flag-only？**

输入特征（每点，**只读 mask 不读值**；值随机置换后特征逐位不变——冒烟断言）：
1. `δ_t = log(1 + time-since-last-observed)`（GRU-D 核心；t 观测到时为 0；前方无观测时为 t+1）
2. `gap_t = log(1 + 到下个观测的距离)`（前向孪生特征；t 观测到时为 0；后方无观测时为 L−t）
3. `rate_t = 窗口内截至 t 的累计观测率`（obs_rate_so_far）

实现：每特征 [B, L] 经 `model.patch` 补丁化为 [B, NPATCH, 16]，特征主序拼接为 [B, NPATCH, k·16]，
以**零初始化**线性投影加进 patch embedding 输出（S5F miss_proj 做法：`emb += W·feats + b`）。
零初始化参数用 `torch.zeros` 显式构造（不消耗 torch RNG），encode 复制品不引入任何 RNG 调用，
因此三变体与 mb 共享**完全相同的数据流与 dropout 流**——delta/full 与 flag 的差异只可能来自
学出来的特征通路。AR 预测第二段 context 变长（576）时补丁数随之变化，投影维度与序列长无关。

## 2. 变体与训练（预算与 SFT-mb 完全一致）

- `flag`：纯 SFT-mb 复跑（锚点；期望与 `s12_ckpt/s12_mb_*` 逐位一致）
- `delta`：flag + δ_t（k=1，投影 13,056 参数）
- `full`：flag + δ_t + gap + rate（k=3，投影 37,632 参数）

训练配置照抄 S6/S12（不做任何调整）：LoRA r=16/α=32/dropout 0.05 on q/k/v/o（peft，3.54M），
3000 步 × batch 256，AdamW lr 1e-4 + 200 步线性 warmup，grad-clip 1.0，clipped pinball clip=100；
增强 = S6-mb 混合（mcar+block24，p~U(0.05,0.8)，20% clean，输入 50% raw-NaN / 50% linear），
train split = 时间轴前 70%，seed = SEED + DS_IDX（即 mb 种子，变体间数据流逐位相同）。
特征投影与 LoRA 参数进同一个 AdamW（同 lr，与 S12 aux head 的处理一致）。

## 3. 评测网格与指标（S6/S12 协议）

150 测试窗（`s5.load_windows(ds,150,SEED)`），1 mask seed，mnar 秩次确定型，中位数分位，H=96，
**nan 原生填充**。网格：机制 {mcar, block, mnar_high, mnar_extreme} × p {0.1,0.3,0.5,0.7} ×
{ETTh1, ETTm1, weather} × {flag, delta, full}（144 cells）+ clean × {zs, flag, delta, full}（12 cells）。
relMSE = MSE / 配对 zs clean MSE；SFT 变体另存 own-clean-relative（S6 的 in-domain 适应护栏）。
每 cell 存 per-window 的 mse / lvl² / signed bias（S19 式水平-形状分解，用于 mnar 低估分析）。

## 4. 判定规则（预注册）

- **主结局**：配对 per-window 检验（variant − flag，zs-clean-relative 单位），池化
  block+mnar_high+mnar_extreme（150 窗 × 4 档 × 3 数据集 × 3 机制，n=5400）。
  **有效** ⇔ delta 或 full 的相对改善 > 2% **且** 95% CI 不含 0（改善方向）。
- **否则**：SFT-mb 已饱和缺失信号通道——与 S12（recon≈mb）、S15（unmask 对 mcar/block 无效）
  的冗余性结论互洽，构成"输入侧信号已用尽"的第三条独立证据。
- mcar 为次要结局（S12 已示 mb 在 mcar 上饱和；全缺 patch 稀少，特征通路面积小）。
- **水平偏差分解**：mnar_high/mnar_extreme 上比较三变体的 signed level bias 与 lvl² 占比——
  S19 发现 mnar_high 次生伤害 = 预测头低估未来均值，问结构化 δ_t 是否减轻该低估。

## 5. 锚点门与阴性/干净校验

- **锚点门**（±5%，期望逐位）：flag 复跑 cells vs `s12_results.json`——clean（flag vs clean:mb，
  zs vs clean:zs，tol 5%/2%）+ {mcar,block} × 4 档 × 3 数据集 flag:nan vs `{mech}:mb:nan:{rate}`，
  共 30 cells；另对 `s21_flag_*.pt` vs `s12_mb_*.pt` 做张量级 torch.equal 逐位检查。
- **干净校验**：clean:{delta,full}/clean:flag ≈ 1（特征在干净输入上为常数 (0,0,1)，训练期 20%
  clean 样本已含同款常数，属在分布内；预期差异 < 3%）。
- **防泄漏断言**：特征只读 mask（冒烟已过）；正式管线特征在 encode 内从 isnan(context) 计算，
  值永不被读。
- **mnar 预期**：S6/S8/S12 一致表明 mcar+block 增强不能合成 MNAR 被审查掉的信息；mnar 列作
  主结局池的一部分是因为 δ_t/gap 在非随机缺失下取值的分布形状变了（ censored 区域 δ 大且成段），
  理论上这是 GRU-D 特征相对 0/1 flag 真正多出来的信息；若 mnar 上有效而 mcar/block 无效，
  结论表述为"特征帮助的是机制识别而非修补"。

## 6. 结果

### 6.1 锚点门 — PASS（逐位一致，强于预注册的 ±5%/±2%）

- 30/30 cells ratio = 1.0000（clean: zs/flag × 3 ds；{mcar,block}×4 档×3 ds flag:nan，
  对照 `s12_results.json` 的 clean:zs/clean:mb 与 `{mech}:mb:nan:{rate}`）。
- **检查点逐位一致**：`s21_flag_{ds}.pt` vs `s12_ckpt/s12_mb_{ds}.pt` 全部 LoRA 张量
  torch.equal = True（max|diff| = 0.0，3/3 数据集）。
- 最终训练损失逐位复现 S6/S12：25.3121 / 27.7125 / 26.5109（ETTh1/ETTm1/weather）。
- 结论：flag 变体就是 SFT-mb 本锚；delta/full 与其共享逐位相同的数据流+dropout 流，
  唯一差异 = 特征通路。

### 6.2 训练曲线（in-sample，仅作 sanity）

最终 clipped pinball：flag 25.31/27.71/26.51 < delta 24.89/27.55/26.24 < full
24.68/27.38/26.05（三数据集一致单调——特征通路确实在学）；|W| 收敛到 0.76–1.00。

### 6.3 主表：三变体全网格 relMSE（vs 配对 zs clean；dataset-avg）

mcar：

| p | flag | delta | full |
|---|------|-------|------|
| 0.1 | 0.7567 | 0.7615 | 0.7637 |
| 0.3 | 0.7714 | 0.7773 | 0.7790 |
| 0.5 | 0.7785 | 0.7818 | 0.7841 |
| 0.7 | 0.8186 | 0.8064 | 0.8027 |
| avg | 0.7813 | 0.7818 | 0.7824 |

block：

| p | flag | delta | full |
|---|------|-------|------|
| 0.1 | 0.7425 | 0.7443 | 0.7437 |
| 0.3 | 0.7680 | 0.7692 | 0.7674 |
| 0.5 | 0.7701 | 0.7705 | 0.7735 |
| 0.7 | 0.7998 | 0.7960 | 0.7937 |
| avg | 0.7701 | 0.7700 | 0.7696 |

mnar_high：

| p | flag | delta | full |
|---|------|-------|------|
| 0.1 | 0.7890 | 0.7929 | 0.7929 |
| 0.3 | 1.0974 | 1.0934 | 1.0804 |
| 0.5 | 1.7064 | 1.6661 | 1.6269 |
| 0.7 | 3.4384 | 3.3991 | 3.3484 |
| avg | 1.7578 | 1.7379 | 1.7122 |

mnar_extreme：

| p | flag | delta | full |
|---|------|-------|------|
| 0.1 | 0.8559 | 0.8597 | 0.8566 |
| 0.3 | 1.1529 | 1.1526 | 1.1505 |
| 0.5 | 1.3034 | 1.3026 | 1.2931 |
| 0.7 | 1.5917 | 1.5885 | 1.5718 |
| avg | 1.2260 | 1.2259 | 1.2180 |

16-cell 总平均：flag 1.1338，delta 1.1289，full 1.1205。

分数据集 rate-avg（flag / delta / full）：
- mcar：ETTh1 0.9292/0.9220/0.9193；ETTm1 0.9307/0.9383/0.9393；weather 0.4840/0.4850/0.4885
- block：ETTh1 0.8794/0.8780/0.8752；ETTm1 0.9401/0.9385/0.9434；weather 0.4908/0.4935/0.4901
- mnar_high：ETTh1 2.0081/2.0039/**1.9508**；ETTm1 2.2363/2.1779/**2.1553**；weather 1.0289/1.0318/1.0305
- mnar_extreme：ETTh1 1.4885/1.4708/1.4614；ETTm1 1.4764/1.4824/1.4772；weather 0.7129/0.7245/0.7153

### 6.4 配对检验（variant − flag，zs-clean-relative per-window，负=改善）

| 对比 | n | mean diff | 相对改善 | t | 95% CI | win |
|---|---|---|---|---|---|---|
| **delta 主结局（block+mnar）** | 5400 | −0.0111 | **+0.52%** | −2.23 | [−0.0209, −0.0013] | 0.497 |
| **full 主结局（block+mnar）** | 5400 | −0.0357 | **+1.68%** | −5.47 | [−0.0484, −0.0229] | 0.537 |
| delta mcar | 1800 | +0.0005 | −0.04% | +0.08 | [−0.0111, +0.0120] | 0.496 |
| delta block | 1800 | +0.0058 | −0.53% | +1.49 | [−0.0019, +0.0134] | 0.488 |
| delta mnar_high | 1800 | −0.0434 | **+1.36%** | −3.18 | [−0.0701, −0.0166] | 0.496 |
| delta mnar_extreme | 1800 | +0.0043 | −0.21% | +0.92 | [−0.0048, +0.0134] | 0.506 |
| full mcar | 1800 | +0.0127 | −1.08% | +1.73 | [−0.0017, +0.0270] | 0.492 |
| full block | 1800 | +0.0091 | −0.83% | +1.51 | [−0.0028, +0.0211] | 0.494 |
| full mnar_high | 1800 | −0.1084 | **+3.40%** | −6.31 | [−0.1421, −0.0747] | 0.547 |
| full mnar_extreme | 1800 | −0.0077 | +0.37% | −1.14 | [−0.0209, +0.0055] | 0.571 |

**主结局未达预注册阈值**（delta +0.52%、full +1.68%，均 < 2%；CI 均不含 0 但效应量不足）。
全部效应集中在 mnar_high 一个机制上，且呈剂量与速率依赖：

- mnar_high 分档（full vs flag）：p=0.1 flat（−0.18t）；p=0.3 **+3.07%**（t=−3.53）；
  p=0.5 **+6.07%**（t=−5.69，最大）；p=0.7 **+2.83%**（t=−3.46）。CI 在 p≥0.3 均不含 0。
- 分数据集（mnar_high，全档池化）：ETTh1 **+3.49%**（t=−6.37）、ETTm1 **+4.52%**（t=−4.85）、
  weather −0.43%（t=+1.53，反方向，win 0.468）——**ETT 专属，weather 无效**（与 S19 unmask
  增益的空间分布完全一致）。
- 特征数剂量响应（full − delta，mnar_high）：p=0.3/0.5/0.7 = +2.05/+2.76/+2.00%，
  |t| ≥ 3.8——delta 拿一半，full 拿全量，单调。
- own-clean-relative（剥离 in-domain 适应，S6 护栏）下存活：mnar_high p=0.7 own-rel
  flag 4.3238 → delta 4.2647 → full 4.1785；p=0.5：2.3524 → 2.2993 → 2.2418。

### 6.5 水平偏差分解（mnar 低估是否减轻？——否）

mnar_high p=0.7 signed level bias（原单位）：ETTh1 flag −3.934 → delta −3.958 →
full −3.897；ETTm1 −3.290 → −3.290 → −3.255；weather −16.664 → −16.725 → −16.750。
lvl² 占比不变（0.73/0.62/0.38 三变体一致）。mnar_extreme 同样无偏差变化。
**S19 的水平脚手架机制（低估减轻）在 S21 特征通路下不成立**：mnar_high 的 relMSE 增益
不是水平校准带来的——系统性低估原封不动，增益是弥散的小幅误差缩减
（与"特征帮助识别'哪里被审查/下个观测在哪'，而不是恢复被审查值"的解读一致）。

### 6.6 干净校验与代价

- clean relMSE（vs zs clean）：flag 0.8727/0.8924/0.4632；delta 0.8754/0.8962/0.4648；
  full 0.8718/0.9110/0.4673（ETTh1/ETTm1/weather）。
- delta/flag clean 比：1.0030 / 1.0042 / 1.0034（≤0.5%，视为无变化 ✅）。
- full/flag clean 比：0.9990 / **1.0208** / 1.0089——full 在 ETTm1 干净集上付 +2.1%、
  weather +0.9% 的小代价（仍在预注册 <3% 容差内，但非零；特征通路在干净输入上是
  常数偏置，对 ETTm1 有轻微过拟合）。
- mcar/block 上 full 配对为**负**（mcar −1.08%、block −0.83%，CI 含 0 或勉强）：
  特征在非随机缺失之外无正作用且可能微有害。

### 6.7 判定

**主结局未达 → 总体判定"SFT-mb 已饱和"（Outcome B），但带一个机制特异的例外。**

1. 预注册主结局（block+mnar 池化，>2% 且 CI 不含 0）：delta +0.52%、full +1.68%——
   **均未过 2% 阈值**。mcar/block/mnar_extreme 上三变体无差异（paired ≈ 0 或为负）。
   与 S12（recon≈mb）、S15（unmask 对 mcar/block 无效）互洽：输入侧缺失信号在 SFT-mb
   下对随机/块状缺失已饱和，GRU-D 特征加不进新东西——**这是第三条独立证据**。
2. **例外**：mnar_high（高值审查）上 full 有真实、统计稳健、剂量依赖的增益
   （p=0.5 +6.07%、p=0.3 +3.07%、p=0.7 +2.83%；ETT 专属；own-clean 下存活）。
   这正是预注册写下的替代情形："特征帮助的是**机制识别**而非修补"——mnar_high 下
   缺失位置的取值分布形状本身泄露机制信息（长段缺失 ↔ 持续高值段），δ/gap/rate 把
   这种跨 patch 的游程结构直接摆在输入层；0/1 flag 信息量等价但需要网络自己算出
   游程结构。**注意：δ/gap/rate 是 flag 的确定型函数，信息论上零新增；增益纯属
   表征便利（inductive bias）**，这与 GRU-D 原始主张同构。
3. 该例外不改变 mnar_high 的整体画面：cell 仍 ~3.3–4.2× 劣化于 clean（own-rel），
   系统性低估未被校准（bias 不动）；且 full 在 mcar 与 ETTm1/weather 干净集上付小代价。
4. 结合 S20（文本通道惰性）：机制信息走结构化输入确实"能进去"（mnar_high 增益为证），
   但在 bolt 这个尺度上，能换来的只有 mnar_high 这一个机制的个位数百分比——
   被审查的信息本身仍然不可恢复（S5/S6/S8 结论第三次复现）。

## 7. 产物

`run_s21_grud.py`（--tiny/--train/--eval [--anchor-subset]/--anchor/--merge/--summary/
--figure）、`s21_results.json`（anchor 门 + 156 cells 含 per-window mse/lvl²/bias +
配对检验 + 汇总表）、`s21.png`、`s21_ckpt/`（s21_{flag,delta,full}_{ds}.pt × 9，
eval_shard0-7.json，anchor.json）、本 notes。日志：`s21_tiny.log`、
`s21_train_{flag,delta,full}_{ds}.log` × 9、`s21_anchor_eval_shard{0-7}.log`、
`s21_anchor.log`、`s21_eval_shard{0-7}.log`、`s21_merge.log`、`s21_summary.log`、
`s21_figure.log`。运行耗时：flag 3×15min（GPU 0-2 并行）→ 锚点 ~8min（GPU 3）→
delta/full 2×15min 两波（GPU 0-2）→ 评测 8 分片 ~6min（GPU 0-3）；总墙钟约 1h05m。

异常与实现注记：
- 初版 `grud_feats` 把补丁数硬编码为 32——AR 预测第二段 context=576（bolt-base
  context_length=2048 不截断）补丁数 36，冒烟即捕获；改为从 patch 输出动态读形状。
- peft 0.20 的 `get_peft_model` 会给基座模型就地写 `peft_config`，且
  PreTrainedModel.base_model 返回自身——`raw_model` 改用 `isinstance(m, PeftModel)`
  判定（冒烟第二次失败修复）。
- 干净代价：full 在 ETTm1 clean +2.1%（§6.6），已在容差内但如实记录。
- 特征零新增信息（δ/gap/rate 皆为 flag 的确定型函数）——所有"增益"的解读边界见 §6.7-2。
- 未改任何已有文件；未用 git；torch/numpy 未动（仅新装零依赖包？无——本轮零安装）。
