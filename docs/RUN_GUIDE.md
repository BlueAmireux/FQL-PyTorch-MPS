# 如何运行与训练

用途：公开仓库的可执行入口。先完成短检查，再自行决定长训预算；本项目已初步结题，不自动启动新训练。以下命令从仓库根目录执行。

## 1. 独立环境与硬件检查

使用Python3.11。曾验证Python3.11.16 / Torch2.14.1；`requirements.in`是安装约束，lock是原机器精确记录。新机器安装后应保存自己的依赖版本，不能因为安装成功就默认与原运行完全一致。

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 复现/fql/pytorch/requirements.in
python -c "import torch; print(torch.__version__); print('MPS built:', torch.backends.mps.is_built(), 'available:', torch.backends.mps.is_available())"
```

MPS available必须为True才能使用`--device mps`。普通CPU机器使用`--device cpu`。训练GPU和MuJoCo物理仿真是两件事；后者在CPU运行。Linux/Windows/CUDA未作为本发布版实测平台，不能据Mac记录保证其运行。

## 2. 不下载数据先核查模型

```sh
python tools/verify_models.py --device cpu
python tools/verify_results.py
```

前者核对九个公开模型文件/source SHA和固定动作探针；后者只读逐回合CSV，重算展示统计，不进行环境交互。Mac可再执行MPS模型探针。输出passed说明检查范围通过，不意味着重新跑过成功率评估。

## 3. MuJoCo短演示

```sh
python tools/demo.py --model state_seed0_average5 --episodes 3 --output runs/state_demo
python tools/demo.py --model visual_seed613_control --episodes 3 --output runs/visual_demo
python tools/demo.py --model visual_seed613_control --suite shifted --episodes 3 --output runs/visual_shift_demo
```

输出目录必须不存在，防止覆盖证据。默认CPU推理，`--device mps`可让模型前向在MPS上执行。每回合使用独立的CPU动作噪声seed；保存第0回合视频，无论成败。默认新环境seed150000起、动作seed151000起，不是正式结果所用病例。

视觉亮度对照必须保持相同seed/suite，分别选择两模型并使用新输出目录：

```sh
python tools/demo.py --model visual_seed613_control --brightness 0.7 --episodes 3 --output runs/control_dim
python tools/demo.py --model visual_seed613_augmentation --brightness 0.7 --episodes 3 --output runs/aug_dim
```

仅改变像素，不改变真实灯光；正常部署优先control。模型名可换成seed614/615；状态模型为seed0/1/2_average5。不要从几个回合选择最好结果作为正式评分。新工具在发布整理时未重新运行环境回合，原正式评估及固定视频见报告。

## 4. 下载并验证状态训练数据

不需要上传官方大数据；由OGBench官方入口获取。下载约束由当前官方服务决定。

```sh
python tools/download_data.py --data-dir DataSet/fql
```

脚本调用`ogbench.download_datasets(['cube-single-play-v0'], ...)`，核对两个文件SHA，不进行训练。随后训练器再次核查SHA、28/5形状、1M/100k有效transition、reward与mask。

不要改官方数据的轨迹划分，也不要把相邻视频帧随机拆入训练与测试。mask不等于原始所有轨迹terminal：使用FQL的success终止规则。

## 5. 从头1000更新，再重载与独立评估

下面是一个新训练链，与公开最终推理权重无关。`--updates`始终表示这次调用**新增**多少optimizer更新；不是总终点，不是环境步。

```sh
python 复现/fql/pytorch/train.py --run-dir runs/state_seed0 \
  --data-dir DataSet/fql --device mps --seed 0 --updates 1000 \
  --optimizer-impl fused --compile --data-placement device \
  --log-every 100 --checkpoint-every 1000
```

预期新目录包含`config.json`、runtime/summary/train CSV，以及带原SHA标记的0/1000 checkpoint。CPU起步则换`--device cpu --data-placement cpu`，可以不加`--compile`；配置一旦建立，续训保持设备/compile/optimizer/线程数。

```sh
python 复现/fql/pytorch/train.py --run-dir runs/state_seed0 \
  --resume-epoch 1000 --device mps --updates 1000 \
  --optimizer-impl fused --compile --data-placement device \
  --log-every 100 --checkpoint-every 1000
```

这会从1000到2000，而不是从1000训练到1000。完整checkpoint保存网络、target、Adam、CPU/MPS RNG和数据采样器；恢复不能只调用`load_state_dict`。新的输出文件拒绝覆盖。Ctrl+C/停止信号在完整更新后保存，退出后检查summary实际终点，再从该终点恢复。

训练与评估用不同入口：

```sh
python 复现/fql/pytorch/evaluate.py --run-dir runs/state_seed0 --epoch 2000 \
  --episodes 5 --device cpu --video --output-dir runs/state_seed0_eval2000
```

评估不加载官方离线数据，不更新optimizer。五回合仅验证训练→保存→重载→MuJoCo链路；早期0成功也应保留，不能反复改seed直到视频成功。确认链路和耗时后，再计划更长训练。

## 6. 长训预算与记录

历史状态配置是3seed各1M。新训练若采用相同预算，先按当前真实随机batch loop测速估计时间：`剩余更新 / 当前有效更新每秒`，另加保存与独立评估。不能直接用本报告的逐次同步固定batch测速安排全部运行。

建议保持loss/耗时低频记录、10k checkpoint间距支持固定average5，100k正式评估，最后800/900/1000k每点100回合。不要把checkpoint间距、日志间距、评估间距混为一件事。保存完整长链可能占较大磁盘；保留支持报告/恢复/平均的必要快照，其余清理应在核查后由使用者决定。

平均策略仅是actor参数导出，不能续训；新链应自行保留完整CP。发布九个模型来自既有实验，本仓库没有把这条新链与它们拼接。

## 7. 代码文件职责

| 文件 | 作用 |
|---|---|
| `fql_torch/networks.py` | MLP、双Q与flow/one-step actor网络 |
| `fql_torch/agent.py` | 四损失、梯度边界、一次Adam与target时序 |
| `fql_torch/data.py` | 官方SHA/shape/reward/mask核查，随机resident batch与采样器 |
| `fql_torch/checkpoint.py` | 完整CP、原子标记、源码SHA、恢复与动作探针 |
| `fql_torch/runtime.py` | CPU/MPS配置、同步、运行信息与内存计数 |
| `train.py` / `evaluate.py` | 状态训练 / 不更新参数的独立评估 |
| `validate_parity.py` / `validate_resume.py` | 原工程数值/短恢复核查；需要原参考材料，不是新机一键再认证 |
| `visual/student.py` | CNN→9latent，与19proprio拼接进状态actor |
| `visual/common.py` | 视觉环境、状态转换、固定教师/学生CP读取 |
| `visual/prepare_data.py` | 官方状态轨迹重渲染；历史v2规则 |
| `visual/train*.py` | 视觉监督迁移、访问数据重放、种子和亮度分支 |
| `visual/collect*.py` / `evaluate*.py` | 访问状态采集与独立闭环评估/诊断 |
| `tools/portable_models.py` | 公开推理格式的独立加载适配，不改冻结算法 |
| `tools/demo.py` / `verify_models.py` | 无数据MuJoCo演示 / 无环境模型探针 |
| `tools/verify_results.py` / `plot_showcase.py` | 公开CSV统计重算 / compact数据绘图 |
| `upstream/fql/` | 固定官方JAX源码供对照，不参与Torch环境安装 |

冻结核心目录不能随便改字节或加入新Python文件，否则历史checkpoint源码SHA校验会拒绝。发布工具放在独立`tools/`。如研究算法改动，建立新配置/训练链、保留旧版与变化说明，不覆盖本项目结果。

## 8. 视觉训练的边界

视觉源码包含历史阶段入口，但其教师固定为原seed0/960k完整CP，且需要未公开的paired/访问/匹配数据和父模型。因此公开版主要提供视觉推理、原实现审阅、结果复查；不能从九个推理导出反推完整历史训练状态。

若未来重做视觉实验，需要先从自己的状态链建立固定教师，再重渲染、按轨迹切分、采集访问状态、匹配数量/时间覆盖和建立同父分支。任务/数据改变后应形成新实验，而不是把新结果挂到旧SHA和旧置信区间上。
