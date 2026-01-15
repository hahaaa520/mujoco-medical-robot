# 医疗机器人高级仿真系统

基于 MuJoCo 3.4.0 的医疗手术机器人仿真平台，集成传感器系统、域随机化和强化学习框架。
采用 2025-2026 年最新的 MuJoCo 最佳实践。

## 主要特性

### 1. 高级物理仿真
- ✅ 精确的刚体动力学和接触力学
- ✅ 优化的求解器参数（50次迭代 + 20次线搜索）
- ✅ 软组织模拟（患者腹部）
- ✅ 真实的摩擦和阻尼参数

### 2. 传感器系统
- **关节传感器**：6个关节的位置和速度
- **力/力矩传感器**：末端执行器的3轴力和力矩
- **触觉传感器**：钳口的接触检测
- **位姿传感器**：末端执行器的位置和姿态（四元数）
- **惯性传感器**：加速度计和陀螺仪

### 3. 域随机化（Domain Randomization）
基于 MuJoCo Playground 2025 最佳实践：
- 摩擦系数随机化
- 质量随机化
- 关节阻尼随机化
- 执行器增益随机化
- 传感器噪声模拟

### 4. Gymnasium 环境
符合 OpenAI Gym/Gymnasium API 标准：
- 标准化的观察和动作空间
- 可配置的奖励函数
- 支持向量化并行环境
- 自动监控和日志记录

### 5. 强化学习支持
集成 Stable-Baselines3，支持：
- SAC（Soft Actor-Critic）
- TD3（Twin Delayed DDPG）
- PPO（Proximal Policy Optimization）
- 自动检查点和评估
- TensorBoard 可视化

## 技术参考

本项目基于以下最新研究和最佳实践：

1. **触觉传感器仿真**: [HydroelasticTouch (2025)](https://arxiv.org/html/2501.08077v1) - 使用 hydroelastic contact surfaces
2. **Gymnasium-Robotics**: [v1.4.2 (2026)](https://robotics.farama.org/) - 最新强化学习环境
3. **域随机化**: [MuJoCo Playground (2025)](https://arxiv.org/html/2502.08844v1) - 领域最佳实践
4. **MuJoCo 文档**: [官方文档 3.0+](https://mujoco.readthedocs.io/)

## 文件结构

```
医疗/
├── medical_scene_advanced.xml      # 高级场景模型（带传感器）
├── medical_robot_env.py            # Gymnasium 环境实现
├── train_rl.py                     # 强化学习训练脚本
├── run_advanced.py                 # 高级运行脚本
├── requirements.txt                # Python 依赖
└── README.md                       # 本文档

生成的文件/
├── logs/                           # 训练日志和 TensorBoard
├── models/                         # 训练好的模型
└── videos/                         # 录制的视频（可选）
```

## 安装

### 1. 基础安装（仅运行仿真）

```bash
# 安装核心依赖
pip install mujoco>=3.4.0 numpy>=1.24.0

# 或使用 requirements.txt
pip install -r requirements.txt
```

### 2. 完整安装（包含强化学习）

```bash
# 安装所有依赖
pip install -r requirements.txt

# 验证安装
python -c "import mujoco; import gymnasium; print('安装成功!')"
```

## 使用方法

### 方式1：基础交互式仿真

```bash
python run_advanced.py
```

**控制说明**：
- `Q/A`: 关节1（底座旋转）
- `W/S`: 关节2（肩部）
- `E/D`: 关节3（肘部）
- `R/F`: 关节4（腕部）
- `T/G`: 左钳口
- `Y/H`: 右钳口
- `M`: 切换自动/手动模式
- `P`: 打印传感器数据
- `SPACE`: 重置场景
- `ESC`: 退出

**功能**：
- 实时传感器数据显示
- 性能监控（FPS）
- 自动演示模式
- 域随机化（可选）

### 方式2：Gymnasium 环境测试

```bash
python medical_robot_env.py
```

测试 Gymnasium 环境的功能：
- 观察空间和动作空间
- 传感器数据读取
- 奖励函数计算
- 域随机化效果

### 方式3：强化学习训练

#### 训练新模型

```bash
# 使用 SAC 算法训练
python train_rl.py --mode train --algorithm SAC --timesteps 500000 --n_envs 4

# 使用 TD3 算法训练
python train_rl.py --mode train --algorithm TD3 --timesteps 500000 --n_envs 4

# 使用 PPO 算法训练
python train_rl.py --mode train --algorithm PPO --timesteps 1000000 --n_envs 8
```

#### 评估训练好的模型

```bash
python train_rl.py --mode evaluate --model_path models/SAC/best_model.zip --n_episodes 20
```

#### 演示随机策略（基线）

```bash
python train_rl.py --mode demo --n_episodes 5
```

### 方式4：自定义训练脚本

```python
from medical_robot_env import MedicalRobotEnv

# 创建环境
env = MedicalRobotEnv(
    xml_path="medical_scene_advanced.xml",
    enable_domain_randomization=True,
    max_episode_steps=500
)

# 运行环境
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # 你的策略
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

## 高级配置

### 1. 域随机化参数

在 `medical_robot_env.py` 中修改：

```python
randomization_params = {
    'friction': True,      # 摩擦系数随机化
    'damping': True,       # 阻尼随机化
    'mass': True,          # 质量随机化
    'actuator': True       # 执行器随机化
}
```

### 2. 奖励函数调整

在 `MedicalRobotEnv` 类中修改 `reward_weights`：

```python
self.reward_weights = {
    'distance': -1.0,      # 距离惩罚
    'velocity': -0.01,     # 速度惩罚
    'action': -0.001,      # 动作惩罚
    'success': 100.0,      # 成功奖励
    'collision': -10.0     # 碰撞惩罚
}
```

### 3. 传感器噪声

在 `_get_obs()` 方法中调整：

```python
noise = np.random.normal(0, 0.01, obs.shape)  # 调整标准差
```

### 4. XML 模型参数

编辑 `medical_scene_advanced.xml`：

```xml
<!-- 求解器精度 -->
<option timestep="0.002" iterations="50" ls_iterations="20"/>

<!-- 执行器增益 -->
<position name="actuator1" joint="joint1" kp="100" kv="10"/>

<!-- 摩擦参数 -->
<geom friction="1.0 0.5 0.5" condim="3"/>
```

## 性能优化建议

### 1. 多环境并行
使用 SubprocVecEnv 进行并行训练：
```python
n_envs = 8  # 根据 CPU 核心数调整
```

### 2. 降低渲染频率
训练时关闭渲染：
```python
env = MedicalRobotEnv(xml_path="...", render_mode=None)
```

### 3. 调整求解器参数
在 XML 中降低迭代次数（对于 RL 训练）：
```xml
<option iterations="30" ls_iterations="10"/>
```

### 4. 使用 GPU（MJX）
对于大规模并行仿真，考虑使用 MuJoCo XLA (MJX)。

## 训练技巧

### 1. 课程学习
逐步增加任务难度：
```python
# 阶段1：近距离目标
max_distance = 0.2

# 阶段2：远距离目标
max_distance = 0.5

# 阶段3：随机目标
max_distance = 1.0
```

### 2. 预训练
使用模仿学习或行为克隆初始化：
```python
# 收集专家数据
expert_data = collect_demonstrations()

# 预训练策略
pretrain_policy(expert_data)
```

### 3. Hyper-parameter Tuning
使用 Optuna 进行超参数优化：
```bash
pip install optuna
# 运行优化脚本
python optimize_hyperparams.py
```

## 监控和可视化

### TensorBoard
```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 在浏览器打开
http://localhost:6006
```

### 性能指标
- 平均奖励（Episode Reward）
- 成功率（Success Rate）
- 平均到达距离（Average Distance）
- 碰撞次数（Collision Count）

## 常见问题

### Q: 训练不收敛怎么办？
A:
1. 降低学习率
2. 增加批次大小
3. 调整奖励函数权重
4. 检查观察空间是否包含足够信息

### Q: 仿真速度慢？
A:
1. 减少并行环境数量
2. 降低求解器迭代次数
3. 关闭域随机化（仅评估时）
4. 使用更快的 CPU

### Q: Sim-to-Real 转移效果差？
A:
1. 增强域随机化范围
2. 添加更多传感器噪声
3. 增加动作延迟
4. 使用真实机器人数据微调

### Q: 如何添加新的传感器？
A:
1. 在 XML 的 `<sensor>` 部分添加定义
2. 在 `SensorReader` 类中添加读取方法
3. 更新观察空间维度

## 扩展项目

### 1. 多机械臂协作
```xml
<!-- 添加第二个机械臂 -->
<body name="robot_base_2" pos="-1.2 0 0">
  ...
</body>
```

### 2. 视觉反馈
```python
# 添加相机传感器
renderer = mujoco.Renderer(model, height=480, width=640)
rgb_array = renderer.render()
```

### 3. 触觉反馈设备集成
使用 ROS 或其他中间件连接真实的触觉设备。

### 4. VR/AR 界面
集成 Unity 或 Unreal Engine 进行沉浸式可视化。

## 引用

如果本项目对你的研究有帮助，请引用相关论文：

```bibtex
@article{mujoco2025,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  journal={IROS},
  year={2012}
}

@software{stable-baselines3,
  author = {Raffin, Antonin and Hill, Ashley and others},
  title = {Stable-Baselines3},
  year = {2021},
  url = {https://github.com/DLR-RM/stable-baselines3}
}
```

## 参考资源

- [MuJoCo 官方文档](https://mujoco.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [MuJoCo Playground (2025)](https://arxiv.org/html/2502.08844v1)
- [HydroelasticTouch (2025)](https://arxiv.org/html/2501.08077v1)
- [Tactile Sensing Environments](https://github.com/carlosferrazza/tactile_envs)

## 许可证

本项目仅供学习和研究使用。MuJoCo 本身遵循 Apache 2.0 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v2.0 (2026-01-15)
- ✅ 集成最新 MuJoCo 3.4.0
- ✅ 添加完整传感器系统
- ✅ 实现域随机化
- ✅ 创建 Gymnasium 环境
- ✅ 集成强化学习框架
- ✅ 性能优化和代码重构

### v1.0 (2026-01-15)
- ✅ 基础医疗场景
- ✅ 6-DOF 机械臂
- ✅ 键盘控制

---

**开发团队**: Claude Code + 用户协作
**最后更新**: 2026-01-15
**MuJoCo 版本**: 3.4.0
**Python 版本**: 3.8+
