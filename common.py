"""公共工具：注册 Fetch 环境，并自动选择已安装的最新版本号。"""
import gymnasium as gym
import gymnasium_robotics

# gymnasium >= 1.0 需要显式注册；旧版本 import 时已自动注册
if hasattr(gym, "register_envs"):
    gym.register_envs(gymnasium_robotics)


def resolve_env_id(name: str) -> str:
    """'FetchReach' -> 当前安装版本中最新的 id，例如 'FetchReach-v4'。
    不同版本的 gymnasium-robotics 版本号不同，这样可以避免硬编码出错。"""
    prefix = name + "-v"
    candidates = [k for k in gym.registry.keys() if k.startswith(prefix)]
    if not candidates:
        raise ValueError(f"找不到环境 {name}，请确认已安装 gymnasium-robotics 和 mujoco")
    return max(candidates, key=lambda k: int(k.split("-v")[-1]))
