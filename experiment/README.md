# 原实验源码归档

用途：展示实际使用的方法实现与核查逻辑。这里的小型Python源码与私有实验归档的原件**逐字节一致**，保留原相对布局，没有移植、泛化为新pipeline或重跑实验。完整历史训练数据、父模型、原训练日志和全部原始轨迹未公开，因此这些归档脚本不构成开箱可运行的历史重训入口。

公开模型推理和短演示使用[运行指南](../docs/RUN_GUIDE.md)及 `tools/verify_models.py` / `tools/demo.py`；模型都是新制作的便携推理格式，不可直接交给下面旧schema的历史loader。完整结果与边界见[最终报告](../docs/FINAL_REPORT.md)和[复现性说明](../docs/REPRODUCIBILITY.md)。

| 归档文件 | 职责 | 原执行所需额外输入 |
|---|---|---|
| [三seed policy.py](fql_torch_three_seed_average/policy.py) | 从同一训练seed的五个10k快照，CPU float64等权平均one-step Actor参数，转float32并保存/加载v2推理artifact；构建和加载集中在一个文件 | 原完整训练checkpoint、marker、config；加载既有v2 artifact仅需artifact/marker和原核心源码 |
| [三seed evaluate_average.py](fql_torch_three_seed_average/evaluate_average.py) | 固定100回合、相同环境/动作seed评估平均策略，逐回合保存原轨迹与官方success | 旧v2平均artifact及marker；无需离线训练数据 |
| [seed0 build_policy.py](fql_torch_policy_average_v2/build_policy.py) | 开发阶段比较average5/average10，保留v1 schema与canonical配置检查 | 原seed0 910k–1M快照；最终选定average5，不能与后续v2 loader混用 |
| [seed0 evaluate_policy.py](fql_torch_policy_average_v2/evaluate_policy.py) | 旧v1平均策略的固定100回合独立CPU评估 | 旧v1 artifact/marker及对应生成源码 |
| [独立平均核查](fql_torch_policy_average_v2/independent_average_validation.py)、[MPS探针核查](fql_torch_policy_average_v2/independent_mps_probe_validation.py) | 分别检查原平均参数、数据身份与重载探针；不把模型平均伪装成继续RL训练 | 开发/确认原件、原完整checkpoint和冻结记录 |
| [matched/match_data.py](fql_torch_visual_matched/match_data.py) | 依据同初始case中teacher/student的共同前缀，生成计数匹配的图像重放；按case分开训练/验证 | 私有同目录protocol、raw_collection数据与manifest、20k视觉父模型、固定state teacher |
| [temporal/prepare_window.py](fql_torch_visual_temporal_coverage/prepare_window.py) | 保持20个case各自样本计数，按 `rint(linspace(0,M-1,N))` 覆盖整条学生轨迹；teacher样本原样保留 | 原matched训练数据、原完整学生轨迹、原同目录protocol/冻结输入 |
| [temporal/audit_training.py](fql_torch_visual_temporal_coverage/audit_training.py) | 检查父模型、Adam/RNG、教师不变、选帧计数，并在固定heldout噪声下复算动作与XYZ误差 | 原模型/segment、原图像/状态验证数据、原source_freeze与protocol |
| [temporal/audit_evaluation.py](fql_torch_visual_temporal_coverage/audit_evaluation.py) | 核查开发评估、官方success、噪声和动作探针，计算配对统计 | 原evaluation_commands/status/freeze、原模型和replay轨迹 |
| [common_confirmation/audit_evaluation.py](fql_torch_visual_common_confirmation/audit_evaluation.py) | 1200回合共同病例确认，保持每case列中的三模型/两方案一起重采样，复核奖励与执行动作 | 原12组replay与冻结评估记录、6个40k模型 |
| [brightness_diagnostic/audit_evaluation.py](fql_torch_visual_brightness_diagnostic/audit_evaluation.py) | 冻结模型下600回合亮度诊断，核查原图到暗图的精确转换及门槛 | 原replay/input_transform、40k模型、原冻结记录 |
| [brightness/audit_training.py](fql_torch_visual_brightness_augmentation/audit_training.py) | 核查同父模型、同预算的control/augmentation分支、严格恢复与brightness因素抽样 | 原40k父CP、41k/60k分支CP、Adam/RNG与训练数据/记录 |
| [brightness/audit_evaluation.py](fql_torch_visual_brightness_augmentation/audit_evaluation.py)、[summarize.py](fql_torch_visual_brightness_augmentation/summarize.py) | 1800回合独立确认与结果整理；保留暗图收益和正常亮度代价，按既定门槛判定 | 原36组replay、60k模型及protocol；汇总读取已经通过的评估审计 |

原视觉训练实现、图像采集与执行器在 `复现/fql/pytorch/visual/`，本目录补充了原本位于experiment中的关键平均、采样与审计实现。这里部分脚本在模块导入时就读取原记录，不要把它们作为公共API导入。它们以私有实验边界和严格SHA为约束，缺少任何冻结原件都会停止；公开副本没有为了绕过检查而改代码。

公开的[状态协议](../evidence/state/protocol.json)、[视觉共同确认协议](../evidence/visual_temporal/protocol.json)、[亮度协议](../evidence/brightness/protocol.json)和逐回合CSV位于 `evidence/`。这些JSON是明确标注的脱敏副本，来源与原SHA见[provenance_manifest.json](../evidence/provenance_manifest.json)。历史protocol包含本机路径，未重复按原字节上传；不应把公共脱敏协议直接复制为旧脚本的执行配置或宣称冻结SHA仍一致。
