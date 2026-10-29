#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "================================"
echo "医疗机器人仿真 - 快速启动"
echo "================================"

if ! python3 -c "import mujoco; import gymnasium; import numpy" 2>/dev/null; then
    echo "缺少依赖。请先在虚拟环境中执行："
    echo "  python3 -m pip install -r requirements.txt"
    exit 1
fi

echo "所有依赖已安装。"
echo ""
echo "可用模式："
echo "  [1] 基础场景"
echo "  [2] 高级场景（带传感器）"
echo "  [3] Gymnasium 环境测试"
echo "  [4] 强化学习随机策略演示"
echo ""
read -r -p "请选择模式 (1-4): " choice

case "$choice" in
    1) python3 run_medical_scene.py ;;
    2) python3 run_advanced.py ;;
    3) python3 medical_robot_env.py ;;
    4) python3 train_rl.py --mode demo --n_episodes 3 ;;
    *)
        echo "无效选择。"
        exit 1
        ;;
esac
