"""用 SAC(或 TQC) + HER 训练 Fetch 任务，支持多环境并行。

用法:
    python train.py --env FetchReach                         # 几分钟收敛
    python train.py --env FetchReach --no-her                # 对照实验：去掉 HER
    python train.py --env FetchPush --algo tqc               # 单环境，约 1M 步
    python train.py --env FetchPush --algo tqc --n-envs 8    # 8 个环境并行，墙钟时间更短
    python train.py --env FetchPush --algo tqc --n-envs 8 --gradient-steps 4  # 再降低更新比，更快
    tensorboard --logdir logs                                # 看训练曲线
"""
import argparse
import os
import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from common import resolve_env_id

# 超参数参考 rl-baselines3-zoo 中 Fetch 任务的设置
CONFIGS = {
    "FetchReach": dict(
        total_steps=20_000, eval_freq=2_000,
        kwargs=dict(learning_rate=1e-3, buffer_size=1_000_000, batch_size=256,
                    gamma=0.95, learning_starts=1_000),
    ),
    "FetchPush": dict(
        total_steps=1_000_000, eval_freq=10_000,
        kwargs=dict(learning_rate=1e-3, buffer_size=1_000_000, batch_size=2048,
                    gamma=0.95, tau=0.05, learning_starts=1_000,
                    policy_kwargs=dict(net_arch=[512, 512, 512])),
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="FetchReach", choices=list(CONFIGS))
    parser.add_argument("--algo", default="sac", choices=["sac", "tqc"])
    parser.add_argument("--no-her", action="store_true", help="关闭 HER，做对照实验")
    parser.add_argument("--steps", type=int, default=None, help="覆盖默认训练步数（所有环境的总步数）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1, help="并行环境数量")
    parser.add_argument("--gradient-steps", type=int, default=-1,
                        help="每次收集数据后做几次梯度更新；-1 表示等于本次收集的步数（保持原来的更新比）")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    args = parser.parse_args()

    cfg = CONFIGS[args.env]
    env_id = resolve_env_id(args.env)
    run_name = f"{args.env}_{args.algo}_{'noher' if args.no_her else 'her'}_s{args.seed}"
    if args.n_envs > 1:
        run_name += f"_n{args.n_envs}_g{args.gradient_steps}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 多个环境放在不同进程里同时跑仿真；make_vec_env 会自动给每个环境套上 Monitor
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)
    eval_env = Monitor(gym.make(env_id))

    if args.algo == "tqc":
        from sb3_contrib import TQC as Algo   # pip install sb3-contrib
    else:
        Algo = SAC

    her_kwargs = {} if args.no_her else dict(
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,                  # 每条真实经验额外生成 4 条改写目标的经验
            goal_selection_strategy="future",  # 用同一 episode 之后到达的位置作为新目标
        ),
    )

    model = Algo(
        "MultiInputPolicy",   # 观测是字典 (observation / achieved_goal / desired_goal)
        env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        train_freq=1,                         # 每个"向量步"（各环境各走一步）训练一次
        gradient_steps=args.gradient_steps,
        tensorboard_log="logs/tb",
        **her_kwargs,
        **cfg["kwargs"],
    )

    # eval_freq 按"向量步"计数，除以环境数，保证每隔相同的总步数评估一次
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=20,
        eval_freq=max(cfg["eval_freq"] // args.n_envs, 1),
        best_model_save_path=log_dir,
        log_path=log_dir,
        deterministic=True,
    )

    model.learn(total_timesteps=args.steps or cfg["total_steps"],
                callback=eval_cb, tb_log_name=run_name, progress_bar=True)
    model.save(os.path.join(log_dir, "final_model"))
    print(f"\n训练完成，模型保存在 {log_dir}/")


if __name__ == "__main__":   # 多进程环境需要这个保护
    main()