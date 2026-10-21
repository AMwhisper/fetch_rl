"""第 3 步：评估训练好的策略，计算成功率，可选弹窗观看或录制视频。

用法:
    python evaluate.py --env FetchReach --model logs/FetchReach_sac_her_s0/best_model.zip
    python evaluate.py ... --render          # 弹窗观看
    python evaluate.py ... --video out.mp4   # 录制视频（无显示器的服务器先 export MUJOCO_GL=egl）
    python evaluate.py ... --video out.mp4 --fps 10 --video-episodes 3   # 慢放（25 为实际速度）
"""
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from common import resolve_env_id

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="FetchReach")
parser.add_argument("--model", required=True)
parser.add_argument("--algo", default="sac", choices=["sac", "tqc"])
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--render", action="store_true")
parser.add_argument("--video", default=None)
parser.add_argument("--fps", type=int, default=25,
                    help="视频帧率；Fetch 每步 0.04 秒，25 为实际速度，10 约慢 2.5 倍")
parser.add_argument("--video-episodes", type=int, default=5, help="录制前几个回合")
args = parser.parse_args()

render_mode = "human" if args.render else ("rgb_array" if args.video else None)
env = gym.make(resolve_env_id(args.env), render_mode=render_mode)

if args.algo == "tqc":
    from sb3_contrib import TQC as Algo
else:
    Algo = SAC
model = Algo.load(args.model, env=env)  # 带 HER 的模型加载时需要传入 env

successes, frames = [], []
for ep in range(args.episodes):
    obs, info = env.reset(seed=1000 + ep)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if args.video and ep < args.video_episodes:
            frames.append(env.render())
        if terminated or truncated:
            break
    successes.append(info["is_success"])

print(f"成功率: {np.mean(successes):.2%}  ({int(np.sum(successes))}/{args.episodes})")

if args.video:
    import imageio  # pip install "imageio[ffmpeg]"
    imageio.mimsave(args.video, frames, fps=args.fps)
    print(f"视频已保存到 {args.video}")
env.close()