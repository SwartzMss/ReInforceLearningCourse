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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gomoku_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QLearningAgent:
    def __init__(self, board_size: int = 15, ai_color: int = 1):
        self.board_size = board_size
        self.ai_color = ai_color  # 1=black, 2=white
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # 初始化Q网络
        input_size = board_size * board_size
        hidden_size = 256
        output_size = board_size * board_size
        self.q_network = QNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 初始化HTTP客户端
        self.base_url = "http://localhost:8080"
        logging.info(f"初始化AI智能体: 棋盘大小={board_size}, AI颜色={'黑子' if ai_color == 1 else '白子'}")
        
    def get_board_state(self) -> np.ndarray:
        """获取当前棋盘状态"""
        try:
            response = requests.get(f"{self.base_url}/get_board")
            data = response.json()
            logging.info(f"获取棋盘状态: 状态码={response.status_code}, 响应={data}")
            return np.array(data["board"])
        except Exception as e:
            logging.error(f"获取棋盘状态失败: {str(e)}")
            raise
    
    def can_move(self) -> Tuple[bool, int]:
        """检查是否可以移动，并返回当前玩家"""
        try:
            response = requests.get(f"{self.base_url}/can_move")
            data = response.json()
            logging.info(f"检查移动状态: 状态码={response.status_code}, 响应={data}")
            return data["can_move"], data["current_player"]
        except Exception as e:
            logging.error(f"检查移动状态失败: {str(e)}")
            raise
    
    def make_move(self, x: int, y: int, color: int) -> bool:
        """在指定位置落子"""
        try:
            data = {
                "x": int(x),
                "y": int(y),
                "color": int(color)
            }
            logging.info(f"尝试落子: 位置=({x}, {y}), 颜色={'黑子' if color == 1 else '白子'}")
            response = requests.post(f"{self.base_url}/move", json=data)
            result = response.json()
            logging.info(f"落子结果: 状态码={response.status_code}, 响应={result}")
            return result["success"]
        except Exception as e:
            logging.error(f"落子失败: {str(e)}")
            raise
    
    def reset_game(self):
        """重置游戏"""
        try:
            response = requests.post(f"{self.base_url}/reset")
            logging.info(f"重置游戏: 状态码={response.status_code}, 响应={response.text}")
        except Exception as e:
            logging.error(f"重置游戏失败: {str(e)}")
            raise
    
    def board_to_state(self, board: np.ndarray) -> torch.Tensor:
        """将棋盘状态转换为网络输入"""
        return torch.FloatTensor(board.flatten())
    
    def get_valid_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """获取所有合法的移动位置"""
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    valid_moves.append((i, j))
        logging.debug(f"获取合法移动: 共{len(valid_moves)}个位置")
        return valid_moves
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, int]:
        """选择动作（落子位置）"""
        if random.random() < self.epsilon:
            # 探索：随机选择合法动作
            board = state.reshape(self.board_size, self.board_size).numpy()
            valid_moves = self.get_valid_moves(board)
            move = random.choice(valid_moves)
            logging.info(f"随机选择动作: 位置=({move[0]}, {move[1]})")
            return int(move[0]), int(move[1])
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                q_values = self.q_network(state)
                board = state.reshape(self.board_size, self.board_size).numpy()
                valid_moves = self.get_valid_moves(board)
                
                # 将非法动作的Q值设为负无穷
                q_values = q_values.reshape(self.board_size, self.board_size)
                for i in range(self.board_size):
                    for j in range(self.board_size):
                        if (i, j) not in valid_moves:
                            q_values[i][j] = float('-inf')
                
                # 选择Q值最大的动作
                max_idx = int(torch.argmax(q_values).item())
                x, y = int(max_idx // self.board_size), int(max_idx % self.board_size)
                logging.info(f"Q网络选择动作: 位置=({x}, {y}), Q值={q_values[x][y]:.4f}")
                return x, y
    
    def train(self, num_episodes: int = 1000):
        """训练Q-learning智能体"""
        logging.info(f"开始训练: 总回合数={num_episodes}")
        for episode in range(num_episodes):
            logging.info(f"开始第{episode}回合训练")
            self.reset_game()
            state = self.board_to_state(self.get_board_state())
            done = False
            
            while not done:
                # 检查是否可以移动和当前玩家
                can_play, current = self.can_move()
                if not can_play:
                    logging.info("游戏已结束，等待下一回合")
                    break
                    
                if current == self.ai_color:
                    # AI的回合
                    logging.info("AI回合开始")
                    # 选择动作
                    action = self.select_action(state)
                    
                    # 执行动作
                    success = self.make_move(action[0], action[1], self.ai_color)
                    
                    if not success:
                        logging.warning("落子失败，尝试其他位置")
                        continue
                    
                    # 获取新状态和奖励
                    next_state = self.board_to_state(self.get_board_state())
                    reward = self.calculate_reward(next_state)
                    logging.info(f"获得奖励: {reward:.4f}")
                    
                    # 更新Q网络
                    self.update_q_network(state, action, reward, next_state)
                    
                    state = next_state
                    
                    # 检查游戏是否结束
                    response = requests.get(f"{self.base_url}/get_board")
                    if response.json()["winner"] != 0:
                        winner = response.json()["winner"]
                        logging.info(f"游戏结束，获胜方={'黑子' if winner == 1 else '白子'}")
                        done = True
                        break
                    
                    # 等待对方落子
                    logging.info("等待对方落子...")
                    while True:
                        can_play, current = self.can_move()
                        if not can_play:
                            logging.info("游戏已结束")
                            done = True
                            break
                        if current != self.ai_color:
                            time.sleep(0.1)  # 等待对方落子
                        else:
                            break  # 对方已落子，轮到AI
                else:
                    # 对方的回合
                    logging.info("等待对方落子...")
                    time.sleep(0.1)
            
            # 更新探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if episode % 10 == 0:
                logging.info(f"训练进度: 回合={episode}, 探索率={self.epsilon:.4f}")
    
    def calculate_reward(self, state: torch.Tensor) -> float:
        """计算奖励值"""
        try:
            response = requests.get(f"{self.base_url}/get_board")
            winner = response.json()["winner"]
            
            if winner == self.ai_color:
                reward = 1.0
                logging.info("AI获胜，获得正奖励")
            elif winner != 0:
                reward = -1.0
                logging.info("AI失败，获得负奖励")
            else:
                reward = 0.0
                logging.debug("游戏继续，无奖励")
            
            return reward
        except Exception as e:
            logging.error(f"计算奖励失败: {str(e)}")
            raise
    
    def update_q_network(self, state: torch.Tensor, action: Tuple[int, int], 
                        reward: float, next_state: torch.Tensor):
        """更新Q网络"""
        try:
            # 将动作转换为索引
            action_idx = action[0] * self.board_size + action[1]
            
            # 计算目标Q值
            with torch.no_grad():
                next_q_values = self.q_network(next_state)
                max_next_q = torch.max(next_q_values)
                target_q = reward + self.gamma * max_next_q
            
            # 计算当前Q值
            current_q = self.q_network(state)[action_idx]
            
            # 计算损失并更新网络
            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logging.debug(f"更新Q网络: 损失={loss.item():.4f}, 目标Q值={target_q:.4f}")
        except Exception as e:
            logging.error(f"更新Q网络失败: {str(e)}")
            raise

if __name__ == "__main__":
    agent = QLearningAgent(ai_color=1)  # 设置AI执黑子
    agent.train(num_episodes=1000)