#!/bin/bash
# 医疗机器人仿真 - 快速启动脚本

echo "================================"
echo "医疗机器人仿真 - 快速启动"
echo "================================"
echo ""

# 检查依赖
echo "1. 检查依赖..."
python3 -c "import mujoco; import gymnasium; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ⚠ 缺少依赖，正在安装..."
    pip3 install --break-system-packages -r requirements.txt
else
    echo "   ✓ 所有依赖已安装"
fi

echo ""
echo "2. 可用的运行模式:"
echo "   [1] 基础场景（原版）"
echo "   [2] 高级场景（带传感器）"
echo "   [3] Gymnasium 环境测试"
echo "   [4] 强化学习演示"
echo ""
echo -n "请选择模式 (1-4): "
read choice

case $choice in
    1)
        echo ""
        echo "启动基础场景..."
        python3 run_medical_scene.py
        ;;
    2)
        echo ""
        echo "启动高级场景..."
        echo "控制: Q/A W/S E/D R/F T/G Y/H"
        echo "      M=切换模式 P=打印传感器 SPACE=重置"
        python3 run_advanced.py
        ;;
    3)
        echo ""
        echo "测试 Gymnasium 环境..."
        python3 medical_robot_env.py
        ;;
    4)
        echo ""
        echo "启动 RL 随机策略演示..."
        python3 train_rl.py --mode demo --n_episodes 3
        ;;
    *)
        echo "无效选择！"
        exit 1
        ;;
esac
