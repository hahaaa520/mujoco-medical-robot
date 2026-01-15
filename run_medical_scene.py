"""
MuJoCo 医疗场景运行脚本
支持键盘控制医疗机械臂
"""

import mujoco
import mujoco.viewer
import numpy as np
import os


def main():
    # 获取 XML 文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "medical_scene.xml")

    # 检查文件是否存在
    if not os.path.exists(xml_path):
        print(f"错误：找不到场景文件 {xml_path}")
        return

    # 加载模型（读取文件内容后从字符串加载，避免中文路径问题）
    print(f"正在加载医疗场景: {xml_path}")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"加载失败：{e}")
        print("尝试使用备用方法...")
        # 备用方法：直接使用路径
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

    print("场景加载成功！")
    print("\n=== 控制说明 ===")
    print("Q/A: 控制关节1（底座旋转）")
    print("W/S: 控制关节2（肩部）")
    print("E/D: 控制关节3（肘部）")
    print("R/F: 控制关节4（腕部旋转）")
    print("T/G: 控制左侧钳口")
    print("Y/H: 控制右侧钳口")
    print("空格: 重置所有关节")
    print("ESC: 退出")
    print("================\n")

    # 控制参数
    control_speed = 0.5
    control = np.zeros(model.nu)

    # 键盘控制回调
    key_pressed = {}

    def key_callback(keycode):
        key_pressed[keycode] = True

    # 使用 viewer 启动仿真
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 设置初始相机位置
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.8]

        # 主循环
        while viewer.is_running():
            step_start = data.time

            # 处理键盘输入
            control[:] = 0  # 重置控制信号

            # 关节1控制（底座旋转）
            if 81 in key_pressed and key_pressed[81]:  # Q
                control[0] = control_speed
            if 65 in key_pressed and key_pressed[65]:  # A
                control[0] = -control_speed

            # 关节2控制（肩部）
            if 87 in key_pressed and key_pressed[87]:  # W
                control[1] = control_speed
            if 83 in key_pressed and key_pressed[83]:  # S
                control[1] = -control_speed

            # 关节3控制（肘部）
            if 69 in key_pressed and key_pressed[69]:  # E
                control[2] = control_speed
            if 68 in key_pressed and key_pressed[68]:  # D
                control[2] = -control_speed

            # 关节4控制（腕部）
            if 82 in key_pressed and key_pressed[82]:  # R
                control[3] = control_speed
            if 70 in key_pressed and key_pressed[70]:  # F
                control[3] = -control_speed

            # 左钳口控制
            if 84 in key_pressed and key_pressed[84]:  # T
                control[4] = control_speed
            if 71 in key_pressed and key_pressed[71]:  # G
                control[4] = -control_speed

            # 右钳口控制
            if 89 in key_pressed and key_pressed[89]:  # Y
                control[5] = control_speed
            if 72 in key_pressed and key_pressed[72]:  # H
                control[5] = -control_speed

            # 重置（空格键）
            if 32 in key_pressed and key_pressed[32]:  # Space
                mujoco.mj_resetData(model, data)
                print("场景已重置")

            # 清空按键状态
            key_pressed.clear()

            # 应用控制
            data.ctrl[:] = control

            # 执行仿真步
            mujoco.mj_step(model, data)

            # 同步 viewer
            viewer.sync()

            # 控制帧率（~60 FPS）
            while (data.time - step_start) < 1.0/60.0:
                mujoco.mj_step(model, data)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
