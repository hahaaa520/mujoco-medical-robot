# 高级使用与自定义

本文档说明高级场景、Gymnasium 环境、域随机化和强化学习脚本。快速入口见根目录 [README](../README.md)。

## 1. 高级场景

推荐入口：

```bash
python run_advanced.py
```

`medical_scene_advanced.xml` 在基础场景上增加：

- 关节、力／力矩、触觉、位姿、速度与惯性传感器。
- 可配置的求解器迭代、摩擦、阻尼和执行器参数。
- 末端执行器与目标点，供距离计算和奖励函数使用。

运行脚本提供手动／自动模式、传感器文本输出和 FPS 显示。FPS 受本机硬件与渲染负载影响，不作为固定性能承诺。

## 2. Gymnasium 环境

检查环境：

```bash
python medical_robot_env.py
```

在自己的代码中使用：

```python
from medical_robot_env import MedicalRobotEnv

env = MedicalRobotEnv(
    xml_path="medical_scene_advanced.xml",
    enable_domain_randomization=True,
    max_episode_steps=500,
)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

环境实现位于 `medical_robot_env.py`。修改观察量、动作空间或奖励权重时，应同步检查环境测试和训练配置，不要只修改文档。

## 3. 域随机化

可在初始化环境时启用：

```python
env = MedicalRobotEnv(
    xml_path="medical_scene_advanced.xml",
    enable_domain_randomization=True,
    randomization_params={
        "friction": True,
        "damping": True,
        "mass": True,
        "actuator": True,
    },
)
```

随机化用于测试策略对参数变化的鲁棒性，不等于已经证明 Sim-to-Real 迁移效果。评估固定模型时，脚本会关闭域随机化以便比较。

## 4. 训练与评估

安装完整依赖：

```bash
python -m pip install -r requirements.txt
```

训练：

```bash
python train_rl.py --mode train --algorithm SAC --timesteps 500000 --n_envs 4
python train_rl.py --mode train --algorithm TD3 --timesteps 500000 --n_envs 4
python train_rl.py --mode train --algorithm PPO --timesteps 1000000 --n_envs 8
```

评估：

```bash
python train_rl.py --mode evaluate \
  --model_path models/SAC/best_model.zip \
  --n_episodes 20
```

无图形界面环境可在评估时增加 `--no-render`。

随机策略基线：

```bash
python train_rl.py --mode demo --n_episodes 3
```

随机策略只用于观察环境行为，不是训练后模型。

## 5. 输出目录

训练脚本会生成：

```text
logs/       训练日志、评估数据与 TensorBoard 文件
models/     检查点、最佳模型与最终模型
videos/     可选的视频输出
```

这些目录已加入 `.gitignore`，不要提交大型模型、本机日志或运行数据。

## 6. 修改 XML 模型

常见修改位置：

- `<option>`：时间步长、求解器迭代次数。
- `<joint>`：关节范围、阻尼和 armature。
- `<geom>`：摩擦、接触参数、材质和质量相关属性。
- `<position>`：执行器增益和控制范围。
- `<sensor>`：传感器定义。

修改后先加载场景，再运行环境检查：

```bash
python -c "import mujoco; mujoco.MjModel.from_xml_path('medical_scene_advanced.xml')"
python medical_robot_env.py
```

如果检出路径包含中文而直接路径加载失败，可参考 `tools/render_preview.py`，先读取 XML 文本再调用 `from_xml_string`。

## 7. 重新渲染场景封面

```bash
python -m pip install "mujoco==3.4.0" pillow
python tools/render_preview.py
```

输出为 `docs/scene-preview.png`。脚本不需要打开交互式 viewer，并支持包含中文的检出路径。
