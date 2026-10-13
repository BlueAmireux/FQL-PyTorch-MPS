# 固定最终模型：便携推理导出

九个模型由结题时已经冻结的模型导出，没有新增训练、重新选优或修改权重。每个文件保留原模型参数、架构、相对来源路径和SHA、原实现源码SHA，以及用于重载核查的输入/动作探针；删除Adam、训练随机状态、数据采样器和本机绝对路径。它们全部是**推理专用，不能用于严格续训**。源码实现仍在 `复现/fql/pytorch/`，与原冻结实现逐字节一致。

| 名称 | 训练来源 | 用途 |
|---|---|---|
| `state_seed0_average5` / `state_seed1_average5` / `state_seed2_average5` | 各state FQL链960k、970k、980k、990k、1M五个保存点的one-step Actor参数等权平均 | 28维状态输入；这是固定部署处理，不是跨种子平均，也不是可续训checkpoint |
| `visual_seed613_control` / `visual_seed614_control` / `visual_seed615_control` | 各40k全时间窗视觉模型正常续训20k，终点60k | 64×64 RGB、19维机器人本体状态和5维Gaussian噪声输入，5维动作输出 |
| `visual_seed613_augmentation` / `visual_seed614_augmentation` / `visual_seed615_augmentation` | 与control同父模型、同预算的亮度增强20k，终点60k | 特定像素亮度扰动的研究对照，正常图像性能有代价，不推荐全面替换control |

`manifest.json` 给出每份导出的大小、SHA、架构、原件SHA、源码SHA和导出动作误差。九个模型合计33,983,702字节（约32.41MiB）。不同种子的结果见最终报告，展示某个种子不代表全部模型的平均水平。

从仓库根目录验证，无需数据、教师或MuJoCo环境交互：

```sh
python tools/verify_models.py --device cpu
# Apple Silicon且MPS可用时：
python tools/verify_models.py --device mps
```

独立小演示命令如下，输出目录须为新目录。脚本按固定第0回合保存视频，不因结果好坏挑视频。发布整理阶段仅验证过CPU/MPS重载和动作探针，下面这些新工具没有重新执行环境回合；先前正式环境评估与视频见报告和证据目录。

```sh
python tools/demo.py --model state_seed0_average5 --episodes 3 --output runs/state_demo
python tools/demo.py --model visual_seed613_control --episodes 3 --output runs/visual_demo
python tools/demo.py --model visual_seed613_control --suite shifted --episodes 3 --output runs/visual_shift_demo
python tools/demo.py --model visual_seed613_control --brightness 0.7 --episodes 3 --output runs/control_dim_demo
python tools/demo.py --model visual_seed613_augmentation --brightness 0.7 --episodes 3 --output runs/augmentation_dim_demo
```

默认环境seed150000起、动作seed151000起，每回合递增。对比两模型时保留相同seed、suite和brightness。默认CPU推理和MuJoCo仿真；`--device mps`只让模型推理使用MPS，MuJoCo仍使用CPU。state模型不接收图像，因此不允许设置brightness。shifted为定制初始位置压力测试，brightness仅变换实际视觉输入像素，不改变真实灯光或动力学；均须与标准任务分开说明。三回合演示不能作为正式成功率估计。

从完整私有原始归档重新生成这些推理导出：

```sh
python tools/export_models.py --source-root /path/to/original-archive
```

导出工具在目标manifest存在时拒绝覆盖。公开仓库不包含原全部checkpoint，因此导出命令不是从公开九个文件恢复原训练状态的方法。state平均导出没有Critic和BC flow teacher；visual训练使用的state teacher也没有包含在这些推理文件中。
