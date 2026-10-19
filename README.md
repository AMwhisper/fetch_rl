# Fetch 机械臂强化学习练习（SAC / TQC + HER）

为加入 Northeastern Helping Hands Lab（机械臂 manipulation 方向）做准备的入门项目。
在 MuJoCo 仿真中，用 Stable-Baselines3 训练 Fetch 机械臂完成 Reach 和 Push 任务，
理解 goal-conditioned RL、稀疏奖励、SAC / TQC 和 HER 的原理与工程细节。

## 文件结构

```
fetch_rl/
├── common.py          # 注册 Fetch 环境，自动选择已安装的最新版本号
├── explore_env.py     # 第 1 步：查看观测/动作/奖励结构，演示 HER 思路，可视化随机策略
├── train.py           # 第 2/4 步：训练（SAC 或 TQC，可选 HER、多环境并行）
├── evaluate.py        # 第 3 步：评估成功率，弹窗观看或录制视频
├── requirements.txt
└── logs/              # 训练后自动生成（模型 + TensorBoard 日志）
```

## 环境配置

```bash
conda create -n fetch python=3.10 -y
conda activate fetch
pip install -r requirements.txt
```

运行环境：Ubuntu 22.04（实体机）+ RTX 4090。训练开始时应显示 `Using cuda device`。

## 使用方法

### 1. 认识环境

```bash
python explore_env.py --env FetchReach
python explore_env.py --env FetchPush --render --delay 0.15 --episodes 5
```

- `--delay`：渲染时每步暂停的秒数；`--episodes`：播放回合数；播放完按回车关闭窗口。
- MuJoCo 窗口中，鼠标拖动旋转视角，滚轮缩放。

### 2. 训练

```bash
python train.py --env FetchReach                    # SAC + HER，约 2 分钟
python train.py --env FetchReach --no-her           # 消融：去掉 HER
python train.py --env FetchPush --algo tqc --n-envs 8 --gradient-steps 4   # Push 推荐配置，约 50 分钟
```

| 参数 | 说明 |
|---|---|
| `--env` | `FetchReach` / `FetchPush` |
| `--algo` | `sac` / `tqc`（TQC 来自 sb3-contrib） |
| `--no-her` | 关闭 HER |
| `--steps` | 覆盖默认总步数（Reach 2 万，Push 100 万） |
| `--seed` | 随机种子 |
| `--n-envs` | 并行环境数（不超过 CPU 核心数） |
| `--gradient-steps` | 每轮收集后的梯度更新次数；-1 = 与收集步数相同 |
| `--device` | `auto` / `cuda` / `cpu` |

长时间训练建议放在 tmux 中运行：`tmux new -s push`，Ctrl+B 再按 D 离开，`tmux attach -t push` 回来。

### 3. 查看训练曲线

```bash
tensorboard --logdir ~/fetch/fetch_rl/logs     # 浏览器打开 localhost:6006
```

重点看 `eval/success_rate`。对比不同配置的速度时，把横轴切换到 `Relative`（墙钟时间）。

### 4. 评估与可视化

```bash
python evaluate.py --env FetchReach --model logs/FetchReach_sac_her_s0/best_model.zip --render
python evaluate.py --env FetchPush --algo tqc --model logs/FetchPush_tqc_her_s0_n8_g4/best_model.zip --render
python evaluate.py --env FetchReach --model <模型路径> --video reach.mp4
```

TQC 训练的模型评估时必须加 `--algo tqc`。

### 训练产物

模型保存在 `logs/<运行名>/`，运行名格式为 `环境_算法_her或noher_s种子[_n环境数_g更新次数]`：

- `best_model.zip`：评估成功率最高的模型（训练中持续更新，中断也会保留）
- `final_model.zip`：训练完全结束时的模型
- `evaluations.npz`：每次评估的原始记录

TensorBoard 日志在 `logs/tb/` 下。相同参数和 seed 重复运行会覆盖旧模型。

## 核心原理速记

- **任务设定**：goal-conditioned RL。观测是字典（`observation`、`achieved_goal`、`desired_goal`），动作是 4 维（末端 dx/dy/dz + 夹爪）。
- **稀疏奖励**：离目标超过 5cm 每步 −1，否则 0；每回合固定 50 步，到达目标不会提前结束。
- **SAC**：最大熵 off-policy actor-critic，双 Q 网络取最小值抑制高估，α 自动调节（目标熵 = −动作维度 = −4）。
- **HER**：把失败轨迹的目标改写成实际到达的位置，将失败变为成功样本（`future` 策略，`n_sampled_goal=4`，约 80% 样本被改写）。
- **TQC**：分布式 critic（每个 25 个分位数），去掉最高的若干分位数来精细控制高估，在 Push 上明显优于 SAC。
- **γ=0.95**：在 0/−1 奖励下，Q 值范围为 [−20, 0]。

## 实验记录

### FetchReach（SAC + HER，seed 0）

- 约 4000 步开始学会，约 14000 步 eval 成功率稳定在 ~100%。
- `eval/mean_reward` 最终约 −1.5，即平均 1.5 步进入目标范围。
- `actor_loss` 先升后降（峰值约 7000 步），约等于 −Q；`ent_coef` 从 ~0.5 降到 0.004。
- 同一 seed 重复两次，曲线几乎重合，结果可复现。

### FetchPush 训练速度测试（TQC + HER，RTX 4090）

| 配置 | 速度 | 100 万步预计用时 |
|---|---|---|
| 单环境 | — | 约 2 小时 |
| `--n-envs 8` | 157 it/s | 约 1 小时 45 分钟 |
| `--n-envs 8 --gradient-steps 4` | 326 it/s | 约 51 分钟 |

**结论**：瓶颈是梯度更新本身（HER 采样 + CPU→GPU 传输 + 小网络计算，由单个 Python 进程串行执行），不是仿真。GPU 功耗只有约 170W / 450W。训练时间几乎与梯度更新次数成正比；多环境并行只节省了约 20 分钟的仿真时间。

## 常见问题

**`AssertionError`（`set_joint_qpos` 中 `mjJNT_HINGE` 判断失败）**
MuJoCo 版本过新，与 gymnasium-robotics 不兼容。降级：`pip install "mujoco==3.2.7"`，仍不行则用 `3.1.6`。开头关于 AdroitHand 的警告可以忽略。

**TensorBoard 提示 `No dashboards are active for the current data set`**
没有找到日志。用绝对路径启动 `--logdir`；用 `find logs -name "events*"` 确认日志存在；训练刚开始时需要等几百步才会写入。

**训练时显示 `Using cpu device`**
先运行 `nvidia-smi` 确认驱动正常；若报错，用 `sudo ubuntu-drivers autoinstall` 安装推荐驱动后重启。驱动正常时，检查 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`，若为 `False` 或版本号带 `+cpu`，按 pytorch.org 的安装选择器重装 CUDA 版 PyTorch。

**渲染画面中红点和方块"闪现"**
回合重置（每 50 步一次）时目标和物块位置重新随机生成，属于正常现象。

**无显示器的服务器上录视频**
先执行 `export MUJOCO_GL=egl`。

## 下一步计划

- [ ] 跑完 FetchPush 完整训练：`--n-envs 8 --gradient-steps 4`，与单环境基准比较成功率（横轴分别用 Step 和 Relative）
- [ ] 若 g4 效果良好，再测试 `--gradient-steps 2`（预计约 30 分钟）
- [ ] FetchReach 的 HER 消融：`--no-her`
- [ ] 多 seed 实验（seed 0/1/2，可在同一张 GPU 上同时运行）
- [ ] 改超参数观察曲线变化：`gamma`、固定 `ent_coef`、`learning_starts`、`n_sampled_goal`
- [ ] 参考 CleanRL 的 `sac_continuous_action.py`，用 PyTorch 手写 SAC + HER，在 FetchReach 上对齐 SB3 效果
- [ ] 进入下一阶段：Diffusion Policy（Push-T）→ BulletArm + 等变网络
