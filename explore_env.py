"""第 1 步：认识环境。看清观测、动作、奖励的结构，并直观理解 HER 的想法。

用法:
    python explore_env.py --env FetchReach
    python explore_env.py --env FetchPush --render   # 弹窗观看随机动作
    python explore_env.py --env FetchPush --render --delay 0.1 --episodes 5  # 更慢、更多回合
"""
import argparse
import time
import gymnasium as gym
import numpy as np
from common import resolve_env_id

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="FetchReach")
parser.add_argument("--render", action="store_true")
parser.add_argument("--delay", type=float, default=0.05, help="渲染时每步暂停的秒数，越大越慢")
parser.add_argument("--episodes", type=int, default=3, help="渲染时播放几个回合")
args = parser.parse_args()

env_id = resolve_env_id(args.env)
env = gym.make(env_id, render_mode="human" if args.render else None)
print(f"环境: {env_id}\n")

obs, info = env.reset(seed=0)
print("== 观测（字典）==")
for k, v in obs.items():
    print(f"  {k:14s} shape={v.shape}  {np.round(v, 3)}")
print(f"\n== 动作空间 ==\n  {env.action_space}  (dx, dy, dz, 夹爪开合)\n")

# 用随机动作跑 episode（Fetch 每个 episode 固定 50 步）
n_episodes = args.episodes if args.render else 1
for ep in range(n_episodes):
    if ep > 0:
        obs, info = env.reset()
    total_reward, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        steps += 1
        if args.render:
            time.sleep(args.delay)   # 放慢播放速度
        if terminated or truncated:
            break
    print(f"== 随机策略 第 {ep + 1} 回合 ==\n  步数={steps}  总奖励={total_reward}  成功={info['is_success']}")
print("  稀疏奖励：没到达目标每步 -1，到达为 0。随机策略几乎永远是 -50。\n")

# HER 的核心直觉：把"实际到达的位置"当成目标，失败轨迹就变成了成功轨迹
u = env.unwrapped
r_orig = u.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
r_relabel = u.compute_reward(obs["achieved_goal"], obs["achieved_goal"], info)
dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
print("== HER 演示 ==")
print(f"  末端/物体离真实目标 {dist:.3f} m（阈值 0.05 m）-> 原奖励 {r_orig}")
print(f"  把目标改写成实际到达的位置                  -> 新奖励 {r_relabel}")
if args.render:
    input("\n按回车键关闭窗口...")   # 播放结束后窗口保持打开
env.close()