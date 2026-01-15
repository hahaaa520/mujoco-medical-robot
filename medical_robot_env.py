"""
医疗机器人高级仿真环境
集成：传感器系统、域随机化、Gymnasium接口
基于 MuJoCo 2025 最佳实践
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import os


class DomainRandomizer:
    """
    域随机化类 - 基于 MuJoCo Playground 2025 最佳实践
    随机化物理参数和视觉参数以提高 sim-to-real 转移
    """

    def __init__(self, model: mujoco.MjModel, randomize_params: Dict[str, bool]):
        self.model = model
        self.params = randomize_params
        self.default_values = self._store_defaults()

    def _store_defaults(self) -> Dict:
        """存储默认参数值"""
        return {
            'friction': self.model.geom_friction.copy(),
            'damping': self.model.dof_damping.copy(),
            'armature': self.model.dof_armature.copy(),
            'mass': self.model.body_mass.copy(),
        }

    def randomize(self, eps: float = 0.2) -> None:
        """
        执行域随机化

        Args:
            eps: 随机化范围 (默认 ±20%)
        """
        if self.params.get('friction', False):
            # 随机化摩擦系数（以默认值为中心）
            for i in range(self.model.ngeom):
                base_friction = self.default_values['friction'][i, 0]
                self.model.geom_friction[i, 0] = np.random.uniform(
                    base_friction * (1 - eps),
                    base_friction * (1 + eps)
                )

        if self.params.get('damping', False):
            # 随机化关节阻尼
            for i in range(self.model.nv):
                base_damping = self.default_values['damping'][i]
                self.model.dof_damping[i] = np.random.uniform(
                    max(0, base_damping * (1 - eps)),
                    base_damping * (1 + eps)
                )

        if self.params.get('mass', False):
            # 随机化物体质量
            for i in range(1, self.model.nbody):  # 跳过世界体
                base_mass = self.default_values['mass'][i]
                self.model.body_mass[i] = np.random.uniform(
                    base_mass * (1 - eps),
                    base_mass * (1 + eps)
                )

        if self.params.get('actuator', False):
            # 随机化执行器增益
            for i in range(self.model.nu):
                base_armature = self.default_values['armature'][i]
                self.model.dof_armature[i] = np.random.uniform(
                    base_armature * (1 - eps),
                    base_armature * (1 + eps)
                )

    def reset(self) -> None:
        """重置所有参数到默认值"""
        self.model.geom_friction[:] = self.default_values['friction']
        self.model.dof_damping[:] = self.default_values['damping']
        self.model.dof_armature[:] = self.default_values['armature']
        self.model.body_mass[:] = self.default_values['mass']


class SensorReader:
    """
    传感器数据读取类
    支持力传感器、触觉传感器、位置传感器等
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._init_sensor_indices()

    def _init_sensor_indices(self):
        """初始化传感器索引"""
        self.sensor_map = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self.sensor_map[name] = i

    def get_joint_positions(self) -> np.ndarray:
        """获取所有关节位置"""
        positions = []
        for name in ['joint1_pos', 'joint2_pos', 'joint3_pos',
                     'joint4_pos', 'left_finger_pos', 'right_finger_pos']:
            if name in self.sensor_map:
                idx = self.sensor_map[name]
                positions.append(self.data.sensordata[idx])
        return np.array(positions)

    def get_joint_velocities(self) -> np.ndarray:
        """获取所有关节速度"""
        velocities = []
        for name in ['joint1_vel', 'joint2_vel', 'joint3_vel',
                     'joint4_vel', 'left_finger_vel', 'right_finger_vel']:
            if name in self.sensor_map:
                idx = self.sensor_map[name]
                velocities.append(self.data.sensordata[idx])
        return np.array(velocities)

    def get_gripper_force(self) -> np.ndarray:
        """获取夹爪力传感器数据"""
        if 'gripper_force' in self.sensor_map:
            idx = self.sensor_map['gripper_force']
            return self.data.sensordata[idx:idx+3]
        return np.zeros(3)

    def get_gripper_torque(self) -> np.ndarray:
        """获取夹爪力矩传感器数据"""
        if 'gripper_torque' in self.sensor_map:
            idx = self.sensor_map['gripper_torque']
            return self.data.sensordata[idx:idx+3]
        return np.zeros(3)

    def get_touch_sensors(self) -> Dict[str, float]:
        """获取触觉传感器数据"""
        touch_data = {}
        for name in ['left_finger_touch', 'right_finger_touch']:
            if name in self.sensor_map:
                idx = self.sensor_map[name]
                touch_data[name] = self.data.sensordata[idx]
        return touch_data

    def get_gripper_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取夹爪位置和姿态"""
        position = np.zeros(3)
        orientation = np.array([1, 0, 0, 0])  # 四元数

        if 'gripper_position' in self.sensor_map:
            idx = self.sensor_map['gripper_position']
            position = self.data.sensordata[idx:idx+3]

        if 'gripper_orientation' in self.sensor_map:
            idx = self.sensor_map['gripper_orientation']
            orientation = self.data.sensordata[idx:idx+4]

        return position, orientation

    def get_target_position(self) -> np.ndarray:
        """获取目标位置"""
        if 'target_position' in self.sensor_map:
            idx = self.sensor_map['target_position']
            return self.data.sensordata[idx:idx+3]
        return np.zeros(3)

    def get_all_sensor_data(self) -> Dict[str, Any]:
        """获取所有传感器数据"""
        return {
            'joint_positions': self.get_joint_positions(),
            'joint_velocities': self.get_joint_velocities(),
            'gripper_force': self.get_gripper_force(),
            'gripper_torque': self.get_gripper_torque(),
            'touch_sensors': self.get_touch_sensors(),
            'gripper_pose': self.get_gripper_pose(),
            'target_position': self.get_target_position(),
        }


class MedicalRobotEnv(gym.Env):
    """
    医疗机器人 Gymnasium 环境
    任务：控制机械臂末端到达手术目标点
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(
        self,
        xml_path: str,
        enable_domain_randomization: bool = True,
        randomization_params: Optional[Dict] = None,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        # 加载模型
        if not os.path.exists(xml_path):
            xml_path = xml_path.replace('\\', '/')

        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except:
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            self.model = mujoco.MjModel.from_xml_string(xml_content)

        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # 初始化域随机化
        self.enable_dr = enable_domain_randomization
        if randomization_params is None:
            randomization_params = {
                'friction': True,
                'damping': True,
                'mass': True,
                'actuator': True
            }
        self.randomizer = DomainRandomizer(self.model, randomization_params)

        # 初始化传感器读取器
        self.sensor_reader = SensorReader(self.model, self.data)

        # 定义动作空间（6个关节的位置控制）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # 定义观察空间
        obs_dim = 6 + 6 + 3 + 3 + 3 + 2  # 位置 + 速度 + 夹爪位置 + 目标位置 + 力 + 触觉
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.viewer = None

        # 奖励权重
        self.reward_weights = {
            'distance': -1.0,
            'velocity': -0.01,
            'action': -0.001,
            'success': 100.0,
            'collision': -10.0
        }

    def _get_obs(self) -> np.ndarray:
        """获取观察"""
        joint_pos = self.sensor_reader.get_joint_positions()
        joint_vel = self.sensor_reader.get_joint_velocities()
        gripper_pos, _ = self.sensor_reader.get_gripper_pose()
        target_pos = self.sensor_reader.get_target_position()
        gripper_force = self.sensor_reader.get_gripper_force()
        touch = self.sensor_reader.get_touch_sensors()
        touch_array = np.array([touch.get('left_finger_touch', 0),
                                touch.get('right_finger_touch', 0)])

        obs = np.concatenate([
            joint_pos, joint_vel, gripper_pos, target_pos,
            gripper_force, touch_array
        ])

        # 添加噪声模拟真实传感器
        noise = np.random.normal(0, 0.01, obs.shape)
        obs += noise

        return obs.astype(np.float32)

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """计算奖励"""
        gripper_pos, _ = self.sensor_reader.get_gripper_pose()
        target_pos = self.sensor_reader.get_target_position()

        # 距离奖励
        distance = np.linalg.norm(gripper_pos - target_pos)
        distance_reward = self.reward_weights['distance'] * distance

        # 速度惩罚（鼓励平滑运动）
        joint_vel = self.sensor_reader.get_joint_velocities()
        velocity_penalty = self.reward_weights['velocity'] * np.sum(np.abs(joint_vel))

        # 动作惩罚（鼓励小动作）
        action_penalty = self.reward_weights['action'] * np.sum(np.square(self.data.ctrl))

        # 成功奖励
        success = distance < 0.02  # 2cm 阈值
        success_reward = self.reward_weights['success'] if success else 0

        # 碰撞检测
        collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 != contact.geom2:  # 非自碰撞
                force = np.linalg.norm(contact.frame[:3])
                if force > 50:  # 力阈值
                    collision = True
                    break
        collision_penalty = self.reward_weights['collision'] if collision else 0

        total_reward = (distance_reward + velocity_penalty +
                       action_penalty + success_reward + collision_penalty)

        info = {
            'distance': distance,
            'success': success,
            'collision': collision,
            'distance_reward': distance_reward,
            'velocity_penalty': velocity_penalty,
            'action_penalty': action_penalty
        }

        return total_reward, success or collision, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        # 域随机化
        if self.enable_dr:
            self.randomizer.randomize(eps=0.2)

        # 重置仿真
        mujoco.mj_resetData(self.model, self.data)

        # 随机化初始关节位置
        if options and options.get('randomize_init', True):
            for i in range(6):
                joint_range = self.model.jnt_range[i]
                self.data.qpos[i] = np.random.uniform(
                    joint_range[0] * 0.5, joint_range[1] * 0.5
                )

        # 前向仿真几步使其稳定
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.current_step = 0

        obs = self._get_obs()
        info = {'sensors': self.sensor_reader.get_all_sensor_data()}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 应用动作（缩放到关节范围）
        action = np.clip(action, -1, 1)
        for i in range(6):
            joint_range = self.model.jnt_range[i]
            self.data.ctrl[i] = action[i] * (joint_range[1] - joint_range[0]) / 2

        # 执行仿真步
        mujoco.mj_step(self.model, self.data)

        # 获取观察和奖励
        obs = self._get_obs()
        reward, terminated, info = self._compute_reward()

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        info['sensors'] = self.sensor_reader.get_all_sensor_data()
        info['step'] = self.current_step

        return obs, reward, terminated, truncated, info

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == 'rgb_array':
            # 返回RGB图像数组
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def test_environment():
    """测试环境"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "medical_scene_advanced.xml")

    print("创建环境...")
    env = MedicalRobotEnv(
        xml_path=xml_path,
        enable_domain_randomization=True,
        render_mode='human'
    )

    print("环境信息:")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")

    print("\n开始测试...")
    obs, info = env.reset()
    print(f"初始观察形状: {obs.shape}")
    print(f"初始传感器数据: {list(info['sensors'].keys())}")

    for episode in range(3):
        obs, info = env.reset()
        total_reward = 0

        print(f"\n=== Episode {episode + 1} ===")

        for step in range(100):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env.render()

            if terminated or truncated:
                print(f"回合结束 - 步数: {step + 1}, 总奖励: {total_reward:.2f}")
                if info.get('success'):
                    print("✓ 成功到达目标!")
                break

    env.close()
    print("\n测试完成!")


if __name__ == "__main__":
    test_environment()
