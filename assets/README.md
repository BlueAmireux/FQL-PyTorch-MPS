# 展示素材

图表来源、哈希与每段视频病例记录见 [showcase_manifest](../evidence/showcase_manifest.json)。图片可通过 `python tools/plot_showcase.py` 从公开compact数据重绘（具体参数见脚本帮助）；统计以报告和逐回合CSV为准。

| 图 | 用途 |
|---|---|
| [state_results.png](state_results.png) | 三种子800/900/1M，末三点score与单终点分开展示 |
| [visual_temporal_coverage.png](visual_temporal_coverage.png) | 三固定模型prefix/full，共同病例配对区间 |
| [brightness_tradeoff.png](brightness_tradeoff.png) | 暗图收益与正常图像代价同时呈现 |
| [mac_benchmark.png](mac_benchmark.png) | 同供电1000更新，吞吐/RSS/全设备驱动代理 |
| [fixed_video_examples.png](fixed_video_examples.png) | 固定ep0成功/失败联系图 |
| [input_brightness_example.png](input_brightness_example.png) | 正常/0.7/0.55实际输入变换示例 |

每组统计图还提供同名SVG。以下均为已有评估的预定第0回合，不因成败更换；选择展示哪些阶段仅为讲述用途，不构成随机样本。

| 视频 | 模型/病例 | 结果 |
|---|---|---|
| [状态500k](videos/state_seed0_500k_fixed_ep0.mp4) | seed0，env3000/action4000，阶段评估 | 成功84步 |
| [状态1M raw](videos/state_seed0_1M_fixed_ep0.mp4) | 同env3000/action4000，阶段评估；不是正式100病例fold | 失败200步 |
| [视觉full40k标准](videos/visual_full40k_seed613_standard_ep0.mp4) | seed613，共同确认env130000/action131000 | 成功59步 |
| [视觉full40k偏移](videos/visual_full40k_seed613_shifted_ep0.mp4) | seed613，共同确认env132000/action133000 | 失败200步 |
| [最终control正常](videos/brightness_control_clean_seed613_ep0.mp4) | seed613，env138000/action139000 | 失败200步 |
| [最终aug正常](videos/brightness_augmented_clean_seed613_ep0.mp4) | 同一病例 | 成功68步 |
| [最终control×0.7](videos/brightness_control_dim07_seed613_ep0.mp4) | 同一病例，暗图输入 | 失败200步 |
| [最终aug×0.7](videos/brightness_augmented_dim07_seed613_ep0.mp4) | 同一病例，暗图输入 | 失败200步 |

最后固定暗图对照两臂均失败，虽然aug在全部暗图病例的成功率明显提高；同一个正常病例aug成功、control失败，也不改变整体正常性能aug下降的结论。单视频不能代表总体平均，更不能拿阶段成功视频代替最终模型成绩。
