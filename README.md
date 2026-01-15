# MuJoCo 医疗场景仿真

基于 MuJoCo 物理引擎构建的医疗手术场景仿真项目，包含手术机器人、手术台、医疗器械等元素。

## 场景组成

### 1. 手术台
- 符合人体工学的手术台设计
- 患者腹部模型
- 手术切口区域标识

### 2. 医疗机械臂（手术机器人）
- **6 个自由度**：
  - 关节 1：底座旋转（360°）
  - 关节 2：肩部关节（240° 范围）
  - 关节 3：肘部关节（280° 范围）
  - 关节 4：腕部旋转（360°）
  - 关节 5-6：双指钳口
- 精确的运动控制
- 金属材质渲染

### 3. 手术器械托盘
包含以下可移动器械：
- 手术刀
- 医用镊子
- 注射器

### 4. 辅助设备
- 监护仪和支架
- 专业照明系统

## 环境要求

### Python 版本
- Python 3.8 或更高版本

### 依赖包
```bash
pip install mujoco
pip install numpy
```

## 安装步骤

1. **安装 MuJoCo**
```bash
pip install mujoco
```

2. **克隆或下载本项目**
确保以下文件在同一目录：
- `medical_scene.xml` - 场景定义文件
- `run_medical_scene.py` - 运行脚本

3. **验证安装**
```bash
python -c "import mujoco; print(mujoco.__version__)"
```

## 使用方法

### 启动场景
```bash
python run_medical_scene.py
```

### 键盘控制

| 按键 | 功能 |
|------|------|
| Q / A | 控制关节1（底座左右旋转）|
| W / S | 控制关节2（肩部上下）|
| E / D | 控制关节3（肘部伸缩）|
| R / F | 控制关节4（腕部旋转）|
| T / G | 控制左侧钳口（张开/闭合）|
| Y / H | 控制右侧钳口（张开/闭合）|
| 空格 | 重置场景到初始状态 |
| ESC | 退出程序 |

### 相机控制
- **鼠标左键拖拽**：旋转视角
- **鼠标右键拖拽**：平移视角
- **鼠标滚轮**：缩放视角
- **双击**：聚焦到点击的物体

## 场景文件说明

### medical_scene.xml
这是 MuJoCo 场景描述文件，使用 XML 格式定义：
- 物体的几何形状和位置
- 材质和纹理
- 关节和执行器
- 物理属性

### 主要组件层次结构
```
worldbody
├── floor（地板）
├── surgery_table（手术台）
│   └── patient_abdomen（患者腹部）
├── robot_base（机械臂基座）
│   └── link1 → link2 → link3 → wrist → gripper（关节链）
├── instrument_tray（器械托盘）
│   ├── scalpel（手术刀）
│   ├── forceps（镊子）
│   └── syringe（注射器）
└── monitor_stand（监护仪）
```

## 自定义场景

### 修改机械臂参数
在 `medical_scene.xml` 中找到相应的 `<joint>` 标签，可以修改：
- `range`：关节运动范围
- `damping`：阻尼系数
- `axis`：旋转轴方向

### 添加新物体
在 `<worldbody>` 中添加新的 `<body>` 元素：
```xml
<body name="new_object" pos="x y z">
  <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
</body>
```

### 调整材质和外观
在 `<asset>` 部分可以定义新的纹理和材质。

## 应用场景

1. **手术机器人研究**
   - 运动学和动力学仿真
   - 控制算法开发和测试
   - 轨迹规划验证

2. **医疗培训**
   - 手术流程模拟
   - 器械操作训练
   - 空间定位练习

3. **强化学习**
   - 自主手术任务学习
   - 精细操作控制
   - 多臂协作研究

## 常见问题

### Q: 运行时提示找不到 mujoco 模块
A: 请确保已安装 mujoco：`pip install mujoco`

### Q: 场景加载失败
A: 检查 `medical_scene.xml` 文件路径是否正确，确保与 Python 脚本在同一目录

### Q: 控制响应延迟
A: 可以在脚本中调整控制频率或降低仿真精度

### Q: 如何录制仿真视频
A: MuJoCo viewer 支持录屏，按 `Ctrl+P` 开始/停止录制

## 扩展建议

1. **添加力反馈**：集成力传感器模拟
2. **增强可视化**：添加手术器械工具提示
3. **多机械臂协作**：添加第二个机械臂
4. **任务自动化**：编写自动化手术流程脚本
5. **VR 集成**：连接 VR 设备实现沉浸式控制

## 技术支持

- MuJoCo 官方文档: https://mujoco.readthedocs.io/
- MuJoCo GitHub: https://github.com/google-deepmind/mujoco

## 许可证

本项目仅供学习和研究使用。

## 版本历史

- v1.0 (2026-01-15)
  - 初始版本
  - 基础手术场景
  - 6-DOF 机械臂
  - 键盘控制接口
