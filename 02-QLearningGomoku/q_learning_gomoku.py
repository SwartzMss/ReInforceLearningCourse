
import numpy as np
import requests
import json
from typing import Tuple, List
import random
import torch
import time
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gomoku_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)


# Q 网络定义：使用 CNN 提取棋盘空间特征，然后用全连接层输出每个位置的 Q 值
class QNetwork(nn.Module):

    def __init__(self, board_size: int, output_size: int):
        super(QNetwork, self).__init__()
        
        # 卷积部分：输入为 [batch, 2通道, 8, 8]，即黑白棋盘两层
        self.conv = nn.Sequential(

            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        conv_out_size = 64 * board_size * board_size
        
        # 全连接部分：提取完空间特征后展平，进一步拟合 Q 值
        self.fc = nn.Sequential(

            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = x.view(-1, 2, 8, 8)  # reshape for Conv2d
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# 智能体类：负责和棋盘交互，训练 Q 网络，选择落子位置
class QLearningAgent:

    def __init__(self, board_size: int = 8, ai_color: int = 1, model_path: str = None):
        self.board_size = board_size
        self.ai_color = ai_color
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # 提高探索下限
        self.epsilon_decay = 0.99

        output_size = board_size * board_size
        self.q_network = QNetwork(board_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logging.info(f"已加载模型 {model_path}, 当前 epsilon={self.epsilon:.4f}")

        self.base_url = "http://localhost:8080"

    def board_to_state(self, board: np.ndarray) -> torch.Tensor:
        
        # 将棋盘状态转换为2通道输入（黑子=1，白子=1）
        black = (board == self.ai_color).astype(np.float32)

        white = (board == 3 - self.ai_color).astype(np.float32)
        state = np.stack([black, white])
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    
    # 主训练循环
    # - 每次游戏循环称为一个 episode
    # - 每 episode 内 AI 与环境交互，不断更新 Q 网络
    def train(self, num_episodes: int = 10000
, save_path: str = "gomoku_model.pth"):
        for episode in range(num_episodes):
            self.reset_game()
            state = self.board_to_state(self.get_board_state())
            done = False
            episode_reward = 0.0

            while not done:
                can_play, current = self.can_move()
                if not can_play:
                    break
                state = self.board_to_state(self.get_board_state())
                if current == self.ai_color:
                    action = self.select_action(state)
                    success, is_over, winner = self.make_move(action[0], action[1], self.ai_color)
                    reward = self.calculate_reward(state, True)
                    if not success:
                        reward -= 0.2
                    episode_reward += reward
                    next_state = self.board_to_state(self.get_board_state())
                    self.update_q_network(state, action, reward, next_state)
                    if is_over:
                        done = True
                        break
                    while True:
                        can_play, current = self.can_move()
                        if not can_play or current == self.ai_color:
                            break
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            logging.info(f"[Ep {episode}] Total reward: {episode_reward:.4f}, Epsilon: {self.epsilon:.4f}")

            if episode % 1000 == 0:
                self.save_model(f"gomoku_model_ep{episode}.pth")

        self.save_model(save_path)
        logging.info(f"训练完成，模型已保存到: {save_path}")

    
    # 奖励函数设计：
    # - 获胜：+1.0
    # - 平局：-0.2
    # - 失败：-1.0
    # - 中间过程根据优势形势给予 ±0.1 奖励
    def calculate_reward(self, state: torch.Tensor
, is_ai_turn: bool) -> float:
        is_over, winner = self.check_game_status()
        if is_over:
            if winner == self.ai_color:
                return 1.0
            elif winner == 3:
                return -0.2
            else:
                return -1.0
        board = state.view(2, self.board_size, self.board_size).detach().cpu().numpy()
        color_board = board[0] if is_ai_turn else board[1]
        enemy_board = board[1] if is_ai_turn else board[0]
        adv = self.count_advantage(color_board)
        danger = self.count_advantage(enemy_board)
        return 0.1 * adv - 0.1 * danger

    def count_advantage(self, board: np.ndarray) -> int:
        count = 0
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in directions:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.check_sequence(board, i, j, dx, dy, 3):
                        count += 1
                    if self.check_sequence(board, i, j, dx, dy, 4):
                        count += 1
        return count

    def check_sequence(self, board: np.ndarray, x: int, y: int, dx: int, dy: int, length: int) -> bool:
        for i in range(length):
            nx, ny = x + i * dx, y + i * dy
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                return False
            if board[nx][ny] != 1:
                return False
        return True

    
    # 执行 Q-Learning 更新：
    # - 计算当前 Q 和目标 Q 之间的损失
    # - 执行梯度下降更新模型参数
    def update_q_network(self,
 state: torch.Tensor, action: Tuple[int, int], reward: float, next_state: torch.Tensor):
        action_idx = action[1] * self.board_size + action[0]
        with torch.no_grad():
            next_q_values = self.q_network(next_state)
            max_next_q = torch.max(next_q_values)
            target_q = reward + self.gamma * max_next_q
        current_q = self.q_network(state)[0][action_idx]
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, state: torch.Tensor) -> Tuple[int, int]:
        board = state.view(2, self.board_size, self.board_size).sum(dim=0).detach().cpu().numpy()
        valid_moves = [(x, y) for y in range(self.board_size) for x in range(self.board_size) if board[y][x] == 0]
        
        # 使用 ε-greedy 策略选择动作：
        # - 以 ε 的概率随机探索
        # - 否则选择 Q 值最大的合法位置
        if random.random() < self.epsilon:

            return random.choice(valid_moves)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)[0].detach().cpu().numpy()
                for y in range(self.board_size):
                    for x in range(self.board_size):
                        idx = y * self.board_size + x
                        if (x, y) not in valid_moves:
                            q_values[idx] = -np.inf
                best_idx = np.argmax(q_values)
                return (best_idx % self.board_size, best_idx // self.board_size)

    def check_game_status(self) -> Tuple[bool, int]:
        response = requests.get(f"{self.base_url}/get_board")
        data = response.json()
        return data["winner"] != 0 or np.all(np.array(data["board"]) != 0), data["winner"]

    def get_board_state(self) -> np.ndarray:
        response = requests.get(f"{self.base_url}/get_board")
        return np.array(response.json()["board"])

    def can_move(self) -> Tuple[bool, int]:
        response = requests.get(f"{self.base_url}/can_move")
        data = response.json()
        return data["can_move"], data["current_player"]

    
    # 尝试执行一次落子操作（通过 HTTP 请求与 Qt 游戏对接）
    def make_move(self, x: int,
 y: int, color: int) -> Tuple[bool, bool, int]:
        is_over, winner = self.check_game_status()
        if is_over:
            return False, True, winner
        data = {
            "x": int(x),
            "y": int(y),
            "color": int(color)
        }
        response = requests.post(f"{self.base_url}/move", json=data)
        result = response.json()
        is_over, winner = self.check_game_status()
        return result["success"], is_over, winner

    
    # 重置棋盘
    def reset_game(self):

        requests.post(f"{self.base_url}/reset")

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

if __name__ == "__main__":
    model_path = "gomoku_model.pth"
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        agent = QLearningAgent(ai_color=2, model_path=model_path if os.path.exists(model_path) else None)
        agent.train(num_episodes=10000, save_path=model_path)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        agent = QLearningAgent(ai_color=1, model_path=model_path)
        agent.test(num_games=100)
