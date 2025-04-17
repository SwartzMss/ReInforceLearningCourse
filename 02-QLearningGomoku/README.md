# 五子棋 Q‑Learning AI – 说明文档

---

## 1 项目简介

本仓库演示如何使用 **PyTorch** 构建并训练一个 Q‑learning 强化学习智能体，让它通过 **HTTP API** 与外部五子棋（Gomoku）游戏引擎交互、自我对弈并学会落子策略。  
整套代码仅依赖一个三层全连接网络，可作为后续引入卷积、经验回放、目标网络等高级技巧的起点。

---

## 2 背景与问题定义

- **五子棋 (Gomoku)**：双方在 15 × 15 棋盘上轮流落子，先连成五子者获胜。  
- **强化学习视角**  
  - **状态 s**：当前棋盘 15×15 的局面（0=空, 1=黑, 2=白）。  
  - **动作 a**：在任意空位落子。  
  - **奖励 r**：终局时 +1（胜）、−1（负）、0（未分胜负）。  
  - **目标**：学习最优行动价值函数 $Q^*(s,a)$，在给定局面下最大化期望回报。

---

## 3 算法原理

### 3.1 Q‑Learning

$$
Q_{\theta}(s,a) \leftarrow Q_{\theta}(s,a) + \alpha \bigl[\, r + \gamma \max_{a'} Q_{\theta}(s',a') - Q_{\theta}(s,a) \bigr]
$$

| 符号 | 含义 |
|------|------|
| $\alpha$ | 学习率 `learning_rate` (0.001) |
| $\gamma$ | 折扣因子 `gamma` (0.99) |
| $s'$ | 执行动作后的下一棋盘状态 |

### 3.2 ε‑贪婪策略

- **探索**：以概率 $\varepsilon$ 随机选择合法落点。  
- **利用**：以概率 $1-\varepsilon$ 选择当前 Q 值最高的落点。  
- 训练过程中 $\varepsilon$ 按 `epsilon_decay=0.995` 指数衰减至 `epsilon_min=0.01`，兼顾探索与收敛。

---

## 4 代码结构

```
├─ README.md              ← 本文件
├─ gomoku_ai.py           ← 主脚本（QNetwork & QLearningAgent）
└─ requirements.txt       ← 依赖清单
```

| 关键类/函数 | 作用 | 备注 |
|-------------|------|------|
| `QNetwork`  | 3×FC 网络，输入/输出维度均为 225 | 可替换为 CNN |
| `QLearningAgent` | 封装训练流程与 HTTP 通信 | 详见下节 |
| `train(num_episodes)` | 自我博弈并更新网络 | 默认 1000 局 |

### 4.1 QLearningAgent API

| 方法 | 描述 |
|------|------|
| `get_board_state()` | GET `/get_board` → 返回 `np.ndarray(15,15)` |
| `can_move()` | GET `/can_move` → 返回轮到谁 & 是否结束 |
| `make_move(x, y, color)` | POST `/move` 落子 |
| `reset_game()` | POST `/reset` 复位棋盘 |
| `select_action(state)` | ε‑贪婪选择动作 |
| `update_q_network(...)` | 反向传播一次 |
| `calculate_reward(state)` | 终局奖励 |

---

## 5 部署与运行

### 5.1 环境依赖

```bash
# Python ≥3.9
pip install -r requirements.txt
# requirements.txt:
# numpy torch requests
```

### 5.2 启动后端五子棋服务

> 本项目假设你已有一个在 **`http://localhost:8080`** 监听的棋盘服务器，> 并提供下列 REST 接口（JSON）：

| 路径 | 方法 | 请求体 | 响应示例 |
|------|------|--------|----------|
| `/reset` | POST | – | `{ "success": true }` |
| `/get_board` | GET | – | `{ "board": [[...]], "winner": 0 }` |
| `/can_move` | GET | – | `{ "can_move": true, "current_player": 1 }` |
| `/move` | POST | `{ "x":7, "y":7, "color":1 }` | `{ "success": true }` |

### 5.3 训练示例

```bash
python q_learning_gomoku.py        # 默认 AI 执黑，自博弈 1000 局
```

所有日志会写入 `gomoku_ai_YYYYMMDD_HHMMSS.log` 并同步打印到终端。

---

## 6 参数说明

| 参数 | 字段 | 默认值 | 作用 |
|------|------|-------|------|
| 学习率 | `learning_rate` | 0.001 | Adam 优化器 LR |
| 折扣因子 | `gamma` | 0.99 | 未来回报权重 |
| 初始探索率 | `epsilon` | 1.0 | ε‑贪婪探索 |
| 最小探索率 | `epsilon_min` | 0.01 | ε 下限 |
| 探索衰减 | `epsilon_decay` | 0.995 | 每局后乘衰减 |
| 隐藏层宽度 | `hidden_size` | 256 | FC 隐层大小 |
| 训练局数 | `num_episodes` | 1000 | 总对局次数 |

> 可按需调整 `board_size`（支持 9/13/15/19）。

---

## 7 结果与日志查看

- **终局胜负** 与 **奖励值** 会在训练日志中输出，可用 `grep` 统计胜率：  
  ```bash
  grep "游戏结束" gomoku_ai_*.log | wc -l         # 总局数
  grep "AI获胜" gomoku_ai_*.log | wc -l            # AI 胜局
  ```
- 若要绘制 **胜率曲线**，可解析 log 或在代码中加入 `matplotlib`。

---

