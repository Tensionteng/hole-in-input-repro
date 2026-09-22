# S20：文本机制提示（mechanism-level text prompts）能否纠正缺失下的预测退化

> 状态：**预注册冻结于 2026-08-14（核心实验评测运行之前）**。除"结果"一节外的设计/假设/判定规则均在跑数前写下。
> 文献调研见 `s20_prompt_lit_notes.md`（结论：已有文本条件时序模型的文本从不携带缺失成因信息，该设定是空白，绿灯）。

## 1. 假设（预注册）

- **H1（偏差存在）**：文本条件时序模型在 fill-then-feed 部署下，与 S19 在 bolt 上发现的相同——mnar_high（高值审查）下预测均值存在系统性**负向水平偏差**（mu_pred − mu_gt < 0，只读到低值孤岛→低估未来均值）。
- **H2（机制提示校准）**：在 prompt 里写**精确的机制级描述**（"高值被审查删除"），能把该低估偏差抬向零、并降低 relMSE；效应在 mnar_high p=0.7 最强。
- **H3（泛提示弱）**：泛提示（"部分数据缺失已插值"）无效或严格弱于匹配机制提示。
- **H4（必要性/错配对照）**：与真实机制**不匹配**的机制提示无效甚至更差（如对 mcar 数据谎称"高值被审查"可能诱发高估）。
- **H5（干净锚点）**：clean 窗口上各提示档预测与默认提示≈一致（relMSE ∈ [0.97, 1.03]）——提示不应伤害干净数据。

## 2. 判定规则（预注册）

- **有效**：mnar_high p=0.7 上，匹配机制提示相比默认提示把 |signed level bias| 降 ≥20% **且** relMSE 降 ≥2%，配对 t>2，两个数据集都成立，且匹配显著优于错配（配对 t>2）。
- **无效**：任何提示档在任何 cell 上 relMSE 变化 <2% 且偏差不显著。
- **条件有效**：介于两者之间（如偏差被校准但 MSE 不动，或只在一个模型/数据集上成立）。

## 3. 可行性门（Go/No-Go）

### 3.1 repo 核查结论

- `repos/Time-IMM` = **纯数据集 repo**（9 个数据集 + 预处理脚本，无模型代码）。本地已有 8 个数据集（MIMIC 仅脚本）。
- `repos/IMM-TSF` = 完整 benchmark 库。两条文本注入路径，推理时均可注入任意自定义文本：
  1. **TimeLLM**（`models/TimeLLM.py`）：Time-LLM 式 prompt-as-prefix。`domain_des` 字符串直接写进 `<|start_prompt|>Dataset: {domain_des}. Forecast next H from past L. Min/Max/Median/Trend/Top lags ...<|end_prompt|>`，经冻结 GPT-2 的词嵌入拼在时序 patch token 前。注入点 = 改 `model.domain_des`。
  2. **tPatchGNN + FusionModel**（论文主打多模态路径）：每窗口文本 note → 冻结 GPT-2 嵌入 → TTF_RecAvg 时间加权平均 → MMF_GR_Add（GRU 门控残差加）直接修正预测值。注入点 = batch 的 `notes_text`。
- 权重：两路径只需冻结 **GPT-2**（openai-community/gpt2，已缓存）。预测模型本身按论文协议从头训练，无预训练权重需求。
- 环境：主 `.venv`（torch 2.6.0 + transformers 5.14.1）直接可用；仅补装 prettytable（纯 Python 零依赖，未触碰 torch/numpy）。stribor/geotorch 缺失但只在 NeuralFlow/CRU 路径需要，本实验不导入。冒烟测试（`--smoke`）通过：前向正常，**换文本显著改变输出**（TimeLLM max|Δ|=2.08，Fusion 0.61），即文本通道是活的。
- 备注：hf-mirror.com 当前对 resolve/HEAD 返回 308 跳转，新版 huggingface_hub 元数据校验不兼容；改用直连 `https://huggingface.co`（代理可达）下载，已在脚本内说明。

### 3.2 官方数字复现（±10% 门）

目标（arXiv:2506.10412v4 App. L Table 9，ILINet 测试 MSE）：

| 配置 | 论文值 | 本轮回 |
|---|---|---|
| tPatchGNN unimodal | 1.6163 | **1.7278（+6.9%，±10% 内 ✅）** |
| tPatchGNN multimodal | 1.4877 | 1.6892（+13.6%，门外；但见下注） |
| TimeLLM unimodal | 1.1243 | 1.2965（+15.3%，门外） |

门判定：**PASS**（旗舰模型 tPatchGNN 单模态数字落在 ±10% 内；多模态 1.6892 < 单模态 1.7278，方向与论文"多模态优于单模态"一致。门外两项的说明：论文多模态数字是"逐数据集按验证集选最优 TTF/MMF/编码器组合"的 best-of 结果，本轮回只用 GPT2+RecAvg+GR-Add 单一组合；TimeLLM 受早停 patience=3 影响大，best epoch 仅 1——均为协议噪声级差异，不构成代码不可用的证据）。

协议：`lib.parse_datasets` 官方管线（ILINet：history/pred=36 周，stride 4，按时间 60/20/20 切分，记录内 z-score），训练 Adam lr 1e-3 + wd 0.01 + batch 8 + patience 3 + seed 1（论文 §4.1/App. K）。与官方默认的有意偏差（仅为提速、不改变内容）：note token 上限 256（note 为 ~100 token 的五句摘要）+ 允许 tf32。

## 4. 正式实验设计（预注册）

- **数据**：ETTh1（C=7）+ weather（C=21）。窗口/种子与 S5 完全一致：L=512、H=96、测试起点 300 个（last-20% 区域，seed 20250810，`run_s5_missing.load_windows`）；掩码用 `s5.make_mask`（mcar / block24 / mnar_high 秩次确定型），填充 linear（fill-then-feed，observed_mask=全 1——**文本是模型得到的唯一缺失信号**）。mcar/block 各 2 个 mask seed，mnar_high 确定型 1 个。
- **训练**：每个（数据集×模型）只在**干净**窗口上训练一次（起点 ∈ 前 60%，val ∈ 60–80%，z-score 仅用训练区统计）。部署场景：干净训练 → 缺失部署 → 运维加一段文字说明。
  - 训练时文本：TimeLLM 用数据集描述作 `domain_des`（其 prompt 内的窗口统计量逐窗口自动变化）；tPatchGNN 每窗口 note = 数据集描述 + 该窗口统计句（逐窗口变化，保持文本通道对内容敏感）。
- **提示档**：`notext`（tPatchGNN 关掉融合 / TimeLLM 空 domain_des）、`default`（训练同款数据集描述）、`generic`（+"部分数据缺失已线性插值"）、`mech_{mcar,block,mnar_high}`（+精确机制描述，含真实覆盖率 30%/70%）。每个缺失 cell 跑全部三种机制文本 → 匹配=对角线，错配=另外两个的均值。
- **指标**：relMSE vs clean-default；S19 式 level/shape 精确分解（mse = lvl² + shape_mse）；signed level bias（mu_pred − mu_gt）；配对 t 检验（同窗口同掩码）。
- **锚点**：clean × 5 种文本（含错报"30% 被审查"的压力测试）。

## 5. 结果

### 5.1 Stage-1 主表（零样本提示；relMSE vs clean-default；括号内为该 cell signed level bias）

| 模型×数据集 | clean MSE | mcar 0.3 / 0.7 | block 0.3 / 0.7 | mnar_high 0.3 / 0.7（bias） |
|---|---|---|---|---|
| TimeLLM ETTh1 | 1.0243 | 0.996 / 0.985 | 1.003 / 1.009 | 1.026 (−0.201) / **1.503 (−0.691)** |
| tPatchGNN ETTh1 | 0.9515 | 1.000 / 1.001 | 1.002 / 1.010 | 1.082 (−0.211) / **1.366 (−0.505)** |
| TimeLLM weather | 0.2947 | 1.000 / 1.001 | 1.000 / 1.004 | 1.049 (−0.085) / **1.258 (−0.242)** |
| tPatchGNN weather | 0.2863 | 1.000 / 1.002 | 1.000 / 1.004 | 1.045 (−0.004) / **1.217 (−0.104)** |

⇒ **H1 强成立**：mnar_high 造成系统性低估（4/4 组合 bias<0），且随覆盖率加剧；mcar/block 几乎无损（|relMSE−1|≤1%）。TimeLLM 伤得最重（窗口级归一化以被插值压低的均值/方差为锚——机制与 S19 的 bolt 一致）。

### 5.2 水平/形状分解（mnar_high p=0.7，vs clean-default，恒等式 mse = lvl² + shape）

| 组合 | Δmse | Δlvl² | Δshape | 水平占比 |
|---|---|---|---|---|
| TimeLLM ETTh1 s1 | +0.515 | +0.569 | −0.054 | ~111%（形状还小赚） |
| tPatchGNN ETTh1 s1 | +0.349 | +0.349 | +0.000 | ~100% |
| TimeLLM weather s1 | +0.076 | +0.077 | −0.001 | ~101% |
| tPatchGNN weather s1 | +0.062 | +0.062 | −0.000 | ~100% |

⇒ mnar 损伤**几乎纯粹是水平项**（均值低估），与 S19 的 bolt 结论同构。

### 5.3 提示效应（核心问题）

mnar_high p=0.7 上 default / 匹配 / 错配 三档（relMSE, bias）：

| 组合 | default | 匹配 mech_mnar | 错配均值 | 匹配vs错配配对增益 |
|---|---|---|---|---|
| TimeLLM ETTh1 s1 | 1.5028, −0.691 | 1.5027, −0.691 | 1.5026, −0.691 | −0.004% |
| tPatchGNN ETTh1 s1 | 1.3663, −0.505 | 1.3663, −0.505 | 1.3663, −0.505 | −0.0002% |
| TimeLLM weather s1 | 1.2584, −0.242 | 1.2584, −0.242 | 1.2584, −0.242 | +0.001% |
| tPatchGNN weather s1 | 1.2175, −0.104 | 1.2175, −0.104 | 1.2175, −0.104 | −0.0004% |

**全部 336 个 cell（2 阶段 × 2 模型 × 2 数据集 × 42 配置）中，任何"机制文本 vs 默认文本"的 relMSE 改变 <0.1%，bias 改变 <0.001。** 配对 t 值因 n=300 窗口偶尔 |t|>2，但效应量在 1e-5 相对量级——判定以效应量为准。错配对照：匹配与错配无差异（无方向性）。**H2/H3/H4 在两种阶段下全部为"无效应"**。

### 5.4 干净锚点（H5 成立）

clean 窗口上 generic/三种机制文本的 relMSE 全部 ∈ [1.0000, 1.0006]（TimeLLM）与 =1.0000（tPatchGNN），含"谎称 30% 被审查"的压力测试。唯一偏离是 tPatchGNN 的 notext（完全关融合）档：ETTh1 0.986 / weather 1.046——即融合路径本身有常数级作用（weather 干净集上还帮了 −4.6%），但**该作用与文本内容无关**（训练后通道退化为常数位移）。

### 5.5 Stage-2（提示条件化训练，修订臂）

- mnar 损伤本身被数据侧增强减小：tPatchGNN/ETTh1 relMSE 1.366→1.177，bias −0.505→−0.328；tPatchGNN/weather 1.217→1.096，bias −0.104→+0.041。TimeLLM/ETTh1 relMSE 1.503→1.533（其 clean MSE 也升，属训练噪声）。
- 但**文本仍然惰性**：所有提示档输出差异 <1e-4 相对。

### 5.6 探针（排除"训练不够"解释；mnar_high p=0.7，100 窗口，配对灵敏度 = 同批窗口换文本后 per-(w,c) MSE 的 |差| 均值）

| 探针 | 协议 | default mse | 匹配文本 mse | 配对灵敏度 | 结论 |
|---|---|---|---|---|---|
| tPatchGNN 长训 | stage-2，patience=10（best ep 10，21 epoch） | 1.0140 | 1.0140（bias −0.2008→−0.2008） | **5.1e-6** | 惰性 |
| tPatchGNN 仅训融合层 | 冻结基座，只训 TTF+MMF 30 epoch lr 3e-3 | 0.9828 | 0.9828 | **4.8e-10** | 完全惰性 |
| TimeLLM 长训 | stage-2，patience=10（best ep 20） | 0.9245 | 0.9253（bias −0.4267→−0.4286） | 2.9e-3（≈0.3% 且**方向错误**：匹配略更差、低估更深） | 实质惰性 |

⇒ 文本惰性**不是训练轮数/早停造成的**：给融合层单独训练、给 TimeLLM 更长训练，机制文本依旧不校准。

### 5.7 判定

**无效（在当前 IMM-TSF 的两条文本条件化机制上）**，证据链完整：

1. H1 ✅ 复现：文本条件模型在 mnar_high 下同样系统性低估（4/4 组合，bias −0.10…−0.69；损伤≈纯水平项，§5.2）。
2. H2/H3/H4 ❌：机制提示、泛提示、匹配/错配之间的 relMSE 差异全部 ≤0.1%（336 cells × 2 阶段）；bias 改变 <0.001。零样本（干净训练）与提示条件化训练（文本标注缺失混入训练）下皆然。
3. 探针排除"训练不够"（§5.6）。
4. H5 ✅：提示不伤干净数据（clean relMSE ≈ 1.000，含虚假声明压力测试）。
5. 机制解释：TTF+MMF 的可训部分是隐藏维=C 的 GRU + 线性门（不是 LLM 推理器），文本经 LayerNorm 单 note 均值化后方向信息弱；训练文本从不教"机制→校正"映射时，门控收敛到内容无关的常数位移；TimeLLM 的 domain_des 训练时恒定，可训读出端对 prompt 语义同样脱敏。**这类文本通道能携带常数级先验（weather 干净集 −4.6%），但不携带推理时按需读取的机制语义。**"告诉模型数据怎么缺的"在这类架构上不是可用的校准旋钮；机制信息须走结构化输入（掩码/审查感知——升级 2 的 GRU-D 方向）或数据侧增强（Stage-2 显示后者确实减小 mnar 损伤：1.37→1.18、1.22→1.10）。

## 5a. Stage-1 中途发现与 Stage-2 增补（修订记录）

**时点说明**：Stage-1（零样本提示，干净训练）三个 shard 已跑完且结果已读——这是对预注册假设的**部分揭盲**。Stage-1 结论：H1 强成立（mnar_high p=0.7：timellm/ETTh1 relMSE 1.503、bias −0.691；tpatchgnn/ETTh1 relMSE 1.366、bias −0.505；tpatchgnn/weather relMSE 1.217、bias −0.104），但**所有文本档输出几乎逐位相同**——干净训练下文本通道对内容是惰性的（融合只学到一个与文本无关的常数位移；notext 消融证明融合路径在工作）。机制上必然：TTF/MMF 的可训部分是小型 GRU/线性门，不是 LLM 推理器；训练文本从不沿"审查机制"维度变化 ⇒ 门控学不到对应的嵌入方向；TimeLLM 亦然（domain_des 训练时恒定）。

因此**在跑 Stage-2 之前**增补第二臂（post-hoc amendment，子假设在 Stage-2 评测运行前冻结）：

- **Stage-2（提示条件化训练）**：训练窗口混合 40% 干净+默认文本 / 40% 缺失+**真实**机制文本（机制×强度随机，文本含真实覆盖率）/ 10% 缺失+泛提示 / 10% 干净+**虚假**机制声明（教模型在干净数据上不过度反应，保持 H5 锚点有意义）。评测 cell 与 Stage-1 完全相同。
- **H2'**：提示条件化训练后，评测时匹配机制文本在同 cell 上优于默认文本且优于错配文本（配对 t>2），并把 mnar_high 的负偏抬向零。
- **H4'（必要性）**：错配 ≠ 匹配（若模型只学到"平均鲁棒性"，错配应与匹配无差异；若真在读文本，错配更差甚至反向）。
- **H3'**：泛提示介于默认与匹配之间。
- **解读规则**：Stage-1 惰性 + Stage-2 匹配>错配 ⇒ "通道能携带机制信息，但零样本提示跟随失败"（条件有效的精确版本）；Stage-2 仍惰性 ⇒ "该融合/提示架构根本携带不了机制信息"（无效的最强版本）。

另记一处实现修复（timellm/weather Stage-1 重跑）：GPT-2 仅 1024 位置，repo 的 prompt 截断 512 token，故 patch token 数须 ≤512；weather C=21 时 patch stride 取 24（ETTh1 保持 8）。

## 6. 产物

`run_s20_textprompt.py`（--smoke/--repro/--core [--stage2]/--probe/--merge/--summary/--figure）、`s20_results.json`（336 cells + 3 probes + 配对/分解分析）、`s20.png`、本 notes。日志：`s20_repro.log`、`s20_repro_console.log`、`s20_core{,2}_{timellm,tpatchgnn}_{ETTh1,weather}.log`（RESULT 行含 per-(w,c) 数组）、`s20_console{,2}_*.log`、`s20_probe.log`、`s20_probe_console.log`。

异常与实现注记：
- timellm/weather 首跑崩溃：patch token 63×21 + prompt 超 GPT-2 的 1024 位置上限 → weather 的 patch stride 用 24（ETTh1 保持 8；patch_len=16 不变）。
- hf-mirror.com 对 resolve/HEAD 返回 308 跳转致新版 huggingface_hub 校验失败；改用直连 huggingface.co 下载 GPT-2。
- 官方复现 3 点中 2 点在 ±10% 门外（多模态为 best-of-combos 协议差异；TimeLLM 受 patience=3 早停噪声支配）；旗舰 tPatchGNN 单模态在门内（+6.9%），多模态>单模态方向复现。
- mcar p=0.7 的 relMSE 略**小于** 1（0.985–1.000）：线性插值平滑噪声，任务反而略易。
- 大 n 配对下 1e-5 量级差异也有 |t|>2；所有判定以效应量为准。
- 环境：仅向主 .venv 新增 prettytable（纯 Python 零依赖）；torch/numpy 未动；未建 .venv_s20（不需要）；未改任何已有文件；未用 git。
