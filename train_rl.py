"""
医疗机器人强化学习训练脚本
使用 Stable-Baselines3 进行训练
支持：SAC, TD3, PPO 等算法
"""

import os
import argparse
from datetime import datetime
import numpy as np
import gymnasium as gym

try:
    from stable_baselines3 import SAC, TD3, PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, CallbackList
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("警告: stable-baselines3 未安装")
    print("安装命令: pip install stable-baselines3")

from medical_robot_env import MedicalRobotEnv


class TrainingConfig:
    """训练配置"""
    def __init__(self):
        # 环境配置
        self.xml_path = "medical_scene_advanced.xml"
        self.n_envs = 4  # 并行环境数量
        self.max_episode_steps = 500

        # 训练配置
        self.algorithm = "SAC"  # SAC, TD3, PPO
        self.total_timesteps = 500000
        self.learning_rate = 3e-4
        self.batch_size = 256
        self.buffer_size = 100000

        # 域随机化
        self.enable_domain_randomization = True
        self.dr_params = {
            'friction': True,
            'damping': True,
            'mass': True,
            'actuator': True
        }

        # 保存配置
        self.save_freq = 10000
        self.eval_freq = 5000
        self.n_eval_episodes = 10

        # 输出目录
        self.log_dir = f"logs/{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_dir = f"models/{self.algorithm}"


def make_env(config: TrainingConfig, rank: int = 0):
    """创建环境工厂函数"""
    def _init():
        env = MedicalRobotEnv(
            xml_path=config.xml_path,
            enable_domain_randomization=config.enable_domain_randomization,
            randomization_params=config.dr_params,
            max_episode_steps=config.max_episode_steps
        )
        env = Monitor(env, filename=f"{config.log_dir}/monitor_{rank}.csv")
        return env
    return _init


def train(config: TrainingConfig):
    """训练函数"""
    if not SB3_AVAILABLE:
        print("错误: 需要安装 stable-baselines3")
        return

    # 创建目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"开始训练 - {config.algorithm}")
    print(f"{'='*60}")
    print(f"并行环境数量: {config.n_envs}")
    print(f"总时间步数: {config.total_timesteps}")
    print(f"域随机化: {config.enable_domain_randomization}")
    print(f"日志目录: {config.log_dir}")
    print(f"{'='*60}\n")

    # 创建向量化环境
    if config.n_envs > 1:
        env = SubprocVecEnv([make_env(config, i) for i in range(config.n_envs)])
    else:
        env = DummyVecEnv([make_env(config, 0)])

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(config, 999)])

    # 创建回调
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq // config.n_envs,
        save_path=config.model_dir,
        name_prefix=config.algorithm
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.model_dir,
        log_path=config.log_dir,
        eval_freq=config.eval_freq // config.n_envs,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # 创建模型
    if config.algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=config.log_dir
        )
    elif config.algorithm == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            verbose=1,
            tensorboard_log=config.log_dir
        )
    elif config.algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=2048 // config.n_envs,
            batch_size=config.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=config.log_dir
        )
    else:
        raise ValueError(f"不支持的算法: {config.algorithm}")

    # 开始训练
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # 保存最终模型
        final_model_path = f"{config.model_dir}/{config.algorithm}_final.zip"
        model.save(final_model_path)
        print(f"\n训练完成! 模型已保存到: {final_model_path}")

    except KeyboardInterrupt:
        print("\n训练被中断!")
        interrupt_model_path = f"{config.model_dir}/{config.algorithm}_interrupted.zip"
        model.save(interrupt_model_path)
        print(f"模型已保存到: {interrupt_model_path}")

    finally:
        env.close()
        eval_env.close()


def evaluate(model_path: str, n_episodes: int = 10, render: bool = True):
    """评估训练好的模型"""
    if not SB3_AVAILABLE:
        print("错误: 需要安装 stable-baselines3")
        return

    print(f"加载模型: {model_path}")

    # 检测算法类型
    if "SAC" in model_path:
        model = SAC.load(model_path)
    elif "TD3" in model_path:
        model = TD3.load(model_path)
    elif "PPO" in model_path:
        model = PPO.load(model_path)
    else:
        print("无法识别算法类型")
        return

    # 创建评估环境
    config = TrainingConfig()
    env = MedicalRobotEnv(
        xml_path=config.xml_path,
        enable_domain_randomization=False,  # 评估时关闭随机化
        max_episode_steps=config.max_episode_steps,
        render_mode='human' if render else None
    )

    print(f"\n开始评估 - {n_episodes} 回合")
    print(f"{'='*60}")

    episode_rewards = []
    success_count = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if render:
                env.render()

            if terminated or truncated:
                if info.get('success'):
                    success_count += 1
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: 奖励 = {episode_reward:.2f}, "
              f"步数 = {steps}, 成功 = {info.get('success', False)}")

    print(f"\n{'='*60}")
    print(f"评估结果:")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  最佳奖励: {np.max(episode_rewards):.2f}")
    print(f"  成功率: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"{'='*60}")

    env.close()


def demo_random_policy(n_episodes: int = 3):
    """演示随机策略（基线）"""
    config = TrainingConfig()
    env = MedicalRobotEnv(
        xml_path=config.xml_path,
        enable_domain_randomization=False,
        max_episode_steps=config.max_episode_steps,
        render_mode='human'
    )

    print(f"演示随机策略 - {n_episodes} 回合")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        print(f"\n=== Episode {episode + 1} ===")

        while steps < config.max_episode_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            env.render()

            if terminated or truncated:
                break

        print(f"奖励: {episode_reward:.2f}, 步数: {steps}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="医疗机器人强化学习训练")
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'demo'],
                       help='运行模式')
    parser.add_argument('--algorithm', type=str, default='SAC',
                       choices=['SAC', 'TD3', 'PPO'],
                       help='强化学习算法')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='总训练步数')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='并行环境数量')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（用于评估）')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='评估回合数')
    parser.add_argument('--no-render', action='store_true',
                       help='评估时不渲染')

    args = parser.parse_args()

    if args.mode == 'train':
        config = TrainingConfig()
        config.algorithm = args.algorithm
        config.total_timesteps = args.timesteps
        config.n_envs = args.n_envs
        train(config)

    elif args.mode == 'evaluate':
        if args.model_path is None:
            print("错误: 评估模式需要指定 --model_path")
            return
        evaluate(args.model_path, args.n_episodes, not args.no_render)

    elif args.mode == 'demo':
        demo_random_policy(args.n_episodes)


if __name__ == "__main__":
    main()
