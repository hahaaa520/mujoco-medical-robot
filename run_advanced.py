"""
医疗机器人高级运行脚本 - 性能优化版
特性：
- 传感器数据实时显示
- 性能优化渲染
- 键盘和自动控制模式
- 数据记录和回放
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import time
from collections import deque
from medical_robot_env import SensorReader, DomainRandomizer


class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, window_size=100):
        self.fps_history = deque(maxlen=window_size)
        self.last_time = time.time()

    def update(self):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            fps = 1.0 / dt
            self.fps_history.append(fps)
        self.last_time = current_time

    def get_fps(self):
        return np.mean(self.fps_history) if self.fps_history else 0


class InteractiveController:
    """交互式控制器"""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.control_mode = "manual"  # manual, auto, demo
        self.control_speed = 0.5
        self.target_joint_pos = np.zeros(6)
        self.sensor_reader = SensorReader(model, data)

        # 记录初始位置
        self.initial_qpos = data.qpos.copy()

    def handle_keyboard(self, key_pressed: dict):
        """处理键盘输入"""
        control_delta = self.control_speed * 0.02

        # 关节控制
        joint_controls = {
            # 关节1
            81: (0, control_delta),    # Q
            65: (0, -control_delta),   # A
            # 关节2
            87: (1, control_delta),    # W
            83: (1, -control_delta),   # S
            # 关节3
            69: (2, control_delta),    # E
            68: (2, -control_delta),   # D
            # 关节4
            82: (3, control_delta),    # R
            70: (3, -control_delta),   # F
            # 钳口
            84: (4, control_delta),    # T
            71: (4, -control_delta),   # G
            89: (5, control_delta),    # Y
            72: (5, -control_delta),   # H
        }

        for key, (joint_idx, delta) in joint_controls.items():
            if key in key_pressed and key_pressed[key]:
                self.target_joint_pos[joint_idx] = np.clip(
                    self.target_joint_pos[joint_idx] + delta,
                    self.model.jnt_range[joint_idx, 0],
                    self.model.jnt_range[joint_idx, 1]
                )

    def auto_control(self):
        """自动控制 - 演示模式"""
        t = self.data.time
        # 正弦波运动模式
        self.target_joint_pos[0] = 0.5 * np.sin(0.5 * t)
        self.target_joint_pos[1] = 0.3 * np.sin(0.3 * t + 1)
        self.target_joint_pos[2] = 0.4 * np.sin(0.4 * t + 2)
        self.target_joint_pos[3] = 0.6 * np.sin(0.6 * t + 3)

    def reset_position(self):
        """重置到初始位置"""
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0
        self.target_joint_pos[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def update_control(self):
        """更新控制信号"""
        if self.control_mode == "auto":
            self.auto_control()

        # 应用控制（位置控制）
        self.data.ctrl[:6] = self.target_joint_pos


class SensorDisplay:
    """传感器数据显示"""
    def __init__(self, sensor_reader):
        self.sensor_reader = sensor_reader

    def get_display_text(self):
        """获取显示文本"""
        sensors = self.sensor_reader.get_all_sensor_data()

        joint_pos = sensors['joint_positions']
        joint_vel = sensors['joint_velocities']
        gripper_pos, _ = sensors['gripper_pose']
        target_pos = sensors['target_position']
        force = sensors['gripper_force']
        touch = sensors['touch_sensors']

        distance = np.linalg.norm(gripper_pos - target_pos)

        text = [
            "=== 医疗机器人仿真 ===",
            "",
            "关节位置 (度):",
            f"  J1: {np.rad2deg(joint_pos[0]):6.1f}°  J2: {np.rad2deg(joint_pos[1]):6.1f}°",
            f"  J3: {np.rad2deg(joint_pos[2]):6.1f}°  J4: {np.rad2deg(joint_pos[3]):6.1f}°",
            f"  钳口: L={np.rad2deg(joint_pos[4]):5.1f}° R={np.rad2deg(joint_pos[5]):5.1f}°",
            "",
            "末端执行器:",
            f"  位置: [{gripper_pos[0]:6.3f}, {gripper_pos[1]:6.3f}, {gripper_pos[2]:6.3f}]",
            f"  速度: {np.linalg.norm(joint_vel[:3]):.3f} m/s",
            f"  力: {np.linalg.norm(force):.2f} N",
            "",
            "目标:",
            f"  位置: [{target_pos[0]:6.3f}, {target_pos[1]:6.3f}, {target_pos[2]:6.3f}]",
            f"  距离: {distance*1000:.1f} mm",
            "",
            "触觉传感器:",
            f"  左指: {touch.get('left_finger_touch', 0):.3f}",
            f"  右指: {touch.get('right_finger_touch', 0):.3f}",
            "",
            "控制:",
            "  Q/A:关节1  W/S:关节2  E/D:关节3",
            "  R/F:关节4  T/G:左钳  Y/H:右钳",
            "  SPACE:重置  M:切换模式  ESC:退出"
        ]

        return "\n".join(text)


def main():
    # 获取文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "medical_scene_advanced.xml")

    # 检查文件
    if not os.path.exists(xml_path):
        print(f"错误：找不到场景文件 {xml_path}")
        return

    # 加载模型
    print("正在加载高级医疗场景...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        model = mujoco.MjModel.from_xml_string(xml_content)

    data = mujoco.MjData(model)

    print("场景加载成功！")
    print(f"传感器数量: {model.nsensor}")
    print(f"执行器数量: {model.nu}")

    # 初始化组件
    controller = InteractiveController(model, data)
    sensor_display = SensorDisplay(controller.sensor_reader)
    performance = PerformanceMonitor()

    # 域随机化（可选）
    enable_dr = False
    if enable_dr:
        dr_params = {'friction': True, 'damping': True, 'mass': False, 'actuator': False}
        randomizer = DomainRandomizer(model, dr_params)
        randomizer.randomize(eps=0.1)
        print("域随机化已启用")

    print("\n" + "="*50)
    print("使用说明:")
    print("  Q/A, W/S, E/D, R/F - 控制前4个关节")
    print("  T/G, Y/H - 控制左右钳口")
    print("  SPACE - 重置场景")
    print("  M - 切换控制模式（手动/自动）")
    print("  P - 打印传感器数据")
    print("  ESC - 退出")
    print("="*50 + "\n")

    # 键盘状态
    key_pressed = {}

    def key_callback(keycode):
        key_pressed[keycode] = True

        # 特殊键处理
        if keycode == 32:  # Space - 重置
            controller.reset_position()
            print("场景已重置")

        elif keycode == 77 or keycode == 109:  # M - 切换模式
            modes = ["manual", "auto"]
            current_idx = modes.index(controller.control_mode)
            controller.control_mode = modes[(current_idx + 1) % len(modes)]
            print(f"控制模式: {controller.control_mode}")

        elif keycode == 80 or keycode == 112:  # P - 打印传感器
            print("\n" + sensor_display.get_display_text())

    # 启动仿真
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 设置相机
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.8]

        # 主循环
        frame_count = 0
        print_interval = 60  # 每60帧打印一次

        while viewer.is_running():
            step_start = data.time

            # 更新控制
            controller.handle_keyboard(key_pressed)
            controller.update_control()

            # 清空按键状态
            key_pressed.clear()

            # 仿真步进
            mujoco.mj_step(model, data)

            # 更新viewer
            viewer.sync()

            # 性能监控
            performance.update()

            # 定期打印信息
            frame_count += 1
            if frame_count % print_interval == 0:
                fps = performance.get_fps()
                print(f"FPS: {fps:.1f} | 模式: {controller.control_mode} | "
                      f"时间: {data.time:.1f}s", end='\r')

            # 帧率控制 (60 FPS)
            while (data.time - step_start) < 1.0/60.0:
                mujoco.mj_step(model, data)

    print("\n仿真结束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
