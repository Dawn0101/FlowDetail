# FlowDetail
启发于ICCV2025 DreamRenderer中关于多对象多属性生成时出现的一系列问题：属性漂移、全局不和谐、主体锚点丢失。

## 实验 A：频域能量对齐（Δx(a) 高频占比）

脚本：`experiments/frequency_alignment.py`

该脚本在同一采样步对比一阶更新与二阶（Heun）更新，抽取加速度项 `Δx(a)`，并对其 2D FFT 计算低/中/高频环能量占比，验证“加速度主要集中在高频细节”。

示例：

```bash
python experiments/frequency_alignment.py \
  --prompt "a man wearing a red hat and blue tracksuit is standing in front of a green sports car" \
  --steps 30 --guidance 3.5 --height 768 --width 768
```

## 实验 B：Mask 时序收敛（stage-1 后半段逐步 refine）

脚本：`experiments/mask_convergence.py`

该脚本在 stage-1 的时间段里用每步更新的幅度构造 soft mask（可理解为主体区域的强变化区域），逐步统计与最终 mask 的 IoU 和面积收敛趋势，验证“mask 可以在后半段逐渐变细、稳定”。

示例：

```bash
python experiments/mask_convergence.py \
  --prompt "a man wearing a red hat and blue tracksuit is standing in front of a green sports car" \
  --steps 30 --guidance 3.5 --stage-ratio 0.6 --keep-ratio 0.2
```

## 注意
- 模型加载方式完全对齐 `flux_dev_load.py` 的本地离线加载方式。
- 若要替换模型目录，请使用 `--model-dir`。
