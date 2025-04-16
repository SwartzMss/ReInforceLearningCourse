# Q-Learning 五子棋智能体

本项目实现了一个基于Q-Learning的强化学习智能体，用于玩五子棋游戏。该智能体通过HTTP接口与Qt编写的五子棋游戏进行交互。

## 环境要求

- Python 3.8+
- PyTorch
- NumPy
- Requests

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保Qt五子棋游戏服务器正在运行（默认地址：`localhost:8080`）
2. 运行训练脚本：

```bash
python q_learning_gomoku.py
```

## 项目结构

- `q_learning_gomoku.py`: 主程序文件，包含Q-Learning智能体的实现
- `requirements.txt`: 项目依赖文件

## 主要功能

- 使用深度Q网络（DQN）学习五子棋策略
- 通过HTTP接口与游戏服务器交互
- 支持探索-利用平衡（ε-greedy策略）
- 自动保存和加载模型

## 注意事项

- 确保游戏服务器正在运行
- 训练过程可能需要较长时间
- 可以根据需要调整超参数（学习率、折扣因子等） 