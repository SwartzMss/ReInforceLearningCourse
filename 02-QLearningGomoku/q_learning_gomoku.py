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
    def __init__(self, board_size: int = 8, ai_color: int = 1, model_path: str = None):
        self.board_size = board_size
        self.ai_color = ai_color  # 1=black, 2=white
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # 初始化Q网络
        input_size = board_size * board_size
        hidden_size = 128  # 减小网络规模以适应更小的棋盘
        output_size = board_size * board_size
        self.q_network = QNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 加载预训练模型（如果存在）
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logging.info(f"已加载预训练模型: {model_path}")
        
        # 初始化HTTP客户端
        self.base_url = "http://localhost:8080"
        logging.info(f"初始化AI智能体: 棋盘大小={board_size}, AI颜色={'黑子' if ai_color == 1 else '白子'}")
        
    def save_model(self, path: str):
        """保存模型权重"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        logging.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型权重"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        logging.info(f"模型已从 {path} 加载")
    
    def check_game_status(self) -> Tuple[bool, int]:
        """检查游戏状态，返回(是否结束, 获胜方)
        返回值说明：
        - 是否结束: True表示游戏结束，False表示游戏继续
        - 获胜方: 0=未结束, 1=黑子胜, 2=白子胜, 3=平局
        """
        try:
            response = requests.get(f"{self.base_url}/get_board")
            data = response.json()
            winner = data["winner"]
            
            # 检查棋盘是否已满（平局）
            board = np.array(data["board"])
            if np.all(board != 0):  # 棋盘已满
                return True, 3  # 3表示平局
            
            # 如果winner不为0，说明有玩家获胜
            if winner != 0:
                return True, winner
            
            # 游戏未结束
            return False, 0
        except Exception as e:
            logging.error(f"检查游戏状态失败: {str(e)}")
            raise

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
    
    def make_move(self, x: int, y: int, color: int) -> Tuple[bool, bool, int]:
        """在指定位置落子，返回(是否成功, 游戏是否结束, 获胜方)
        返回值说明：
        - 是否成功: True表示落子成功，False表示落子失败
        - 游戏是否结束: True表示游戏结束，False表示游戏继续
        - 获胜方: 0=未结束, 1=黑子胜, 2=白子胜, 3=平局
        """
        try:
            # 先检查游戏状态
            is_over, winner = self.check_game_status()
            if is_over:
                winner_text = {
                    0: "未结束",
                    1: "黑子",
                    2: "白子",
                    3: "平局"
                }[winner]
                logging.info(f"游戏已结束，状态：{winner_text}")
                return False, True, winner

            data = {
                "x": int(x),
                "y": int(y),
                "color": int(color)
            }
            logging.info(f"尝试落子: 位置=({x}, {y}), 颜色={'黑子' if color == 1 else '白子'}")
            response = requests.post(f"{self.base_url}/move", json=data)
            result = response.json()
            logging.info(f"落子结果: 状态码={response.status_code}, 响应={result}")
            
            # 落子后再次检查游戏状态
            is_over, winner = self.check_game_status()
            return result["success"], is_over, winner
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
        valid_moves: List[Tuple[int,int]] = []
        # i=row, j=col -> 这里反过来存为 (x=col, y=row)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == 0:
                    valid_moves.append((col, row))
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
                can_play, current = self.can_move()
                if not can_play:
                    logging.info("游戏已结束，等待下一回合")
                    break
                
                # 确保获取最新的棋盘状态
                state = self.board_to_state(self.get_board_state())
                    
                if current == self.ai_color:
                    # AI的回合
                    logging.info("AI回合开始")
                    # 选择动作
                    action = self.select_action(state)
                    
                    # 执行动作
                    success, is_over, winner = self.make_move(action[0], action[1], self.ai_color)
                    
                    if not success:
                        logging.warning("落子失败，尝试其他位置")
                        continue
                    
                    # 获取新状态和奖励
                    next_state = self.board_to_state(self.get_board_state())
                    reward = self.calculate_reward(next_state, True)
                    logging.info(f"获得奖励: {reward:.4f}")
                    
                    # 更新Q网络
                    self.update_q_network(state, action, reward, next_state)
                    
                    state = next_state
                    
                    # 如果游戏结束，更新done标志
                    if is_over:
                        winner_text = {
                            0: "未结束",
                            1: "黑子",
                            2: "白子",
                            3: "平局"
                        }[winner]
                        logging.info(f"游戏结束，状态：{winner_text}")
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
                            # 对方已落子，获取最新状态
                            state = self.board_to_state(self.get_board_state())
                            break  # 轮到AI
                else:
                    # 对方的回合
                    logging.info("等待对方落子...")
                    time.sleep(0.1)
            
            # 更新探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if episode % 10 == 0:
                logging.info(f"训练进度: 回合={episode}, 探索率={self.epsilon:.4f}")
    
    def calculate_reward(self, state: torch.Tensor, is_ai_turn: bool) -> float:
        """计算奖励值
        参数说明：
        - state: 当前棋盘状态
        - is_ai_turn: 是否是AI的回合
        
        返回值说明：
        - 1.0: 直接获胜
        - -1.0: 对手直接获胜
        - -0.2: 平局
        - 0.0: 游戏继续
        """
        try:
            is_over, winner = self.check_game_status()
            
            if is_over:
                if winner == self.ai_color:
                    return 1.0  # AI获胜
                elif winner == 3:
                    return -0.2  # 平局
                else:
                    return -1.0  # 对手获胜
            
            return 0.0  # 游戏继续
            
        except Exception as e:
            logging.error(f"计算奖励失败: {str(e)}")
            raise
    
    def check_winning_position(self, board: np.ndarray, color: int) -> bool:
        """检查是否形成必胜局面（下一步可以连成5子）"""
        # 检查所有可能的落子位置
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:  # 空位
                    # 模拟落子
                    board[i][j] = color
                    if self.check_win(board, color):
                        board[i][j] = 0  # 恢复棋盘
                        return True
                    board[i][j] = 0  # 恢复棋盘
        return False
    
    def check_advantage_position(self, board: np.ndarray, color: int) -> bool:
        """检查是否形成优势局面（有多个活三或冲四）"""
        advantage_count = 0
        # 检查所有可能的落子位置
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:  # 空位
                    # 模拟落子
                    board[i][j] = color
                    if self.count_advantage(board, color) >= 2:  # 至少有两个优势
                        board[i][j] = 0  # 恢复棋盘
                        return True
                    board[i][j] = 0  # 恢复棋盘
        return False
    
    def count_advantage(self, board: np.ndarray, color: int) -> int:
        """计算优势数量（活三和冲四的数量）"""
        count = 0
        # 检查所有方向
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in directions:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.check_sequence(board, i, j, dx, dy, color, 3):  # 活三
                        count += 1
                    if self.check_sequence(board, i, j, dx, dy, color, 4):  # 冲四
                        count += 1
        return count
    
    def check_sequence(self, board: np.ndarray, x: int, y: int, dx: int, dy: int, color: int, length: int) -> bool:
        """检查指定位置是否形成指定长度的连续棋子"""
        for i in range(length):
            nx, ny = x + i*dx, y + i*dy
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                return False
            if board[nx][ny] != color:
                return False
        return True
    
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

    def test(self, num_games: int = 100):
        """测试模型性能"""
        logging.info(f"开始测试: 游戏数量={num_games}")
        wins = 0
        losses = 0
        draws = 0
        
        for game in range(num_games):
            logging.info(f"开始第{game}局测试")
            self.reset_game()
            state = self.board_to_state(self.get_board_state())
            done = False
            
            while not done:
                can_play, current = self.can_move()
                if not can_play:
                    break
                
                # 确保获取最新的棋盘状态
                state = self.board_to_state(self.get_board_state())
                    
                if current == self.ai_color:
                    # AI的回合
                    action = self.select_action(state)
                    success, is_over, winner = self.make_move(action[0], action[1], self.ai_color)
                    
                    if not success:
                        if is_over:
                            if winner == 3:  # 平局
                                draws += 1
                                logging.info("平局")
                            elif winner == self.ai_color:
                                wins += 1
                                logging.info("AI获胜")
                            else:
                                losses += 1
                                logging.info("AI失败")
                            done = True
                        continue
                    
                    # 获取新状态
                    next_state = self.board_to_state(self.get_board_state())
                    state = next_state
                    
                    # 如果游戏结束，记录结果
                    if is_over:
                        if winner == 3:  # 平局
                            draws += 1
                            logging.info("平局")
                        elif winner == self.ai_color:
                            wins += 1
                            logging.info("AI获胜")
                        else:
                            losses += 1
                            logging.info("AI失败")
                        done = True
                        break
                    
                    # 等待对方落子
                    logging.info("等待对方落子...")
                    while True:
                        can_play, current = self.can_move()
                        if not can_play:
                            done = True
                            break
                        if current != self.ai_color:
                            time.sleep(0.1)
                        else:
                            # 对方已落子，获取最新状态
                            state = self.board_to_state(self.get_board_state())
                            break
                else:
                    # 对方的回合
                    logging.info("等待对方落子...")
                    time.sleep(0.1)
            
            if not done:
                # 检查最终游戏状态
                is_over, winner = self.check_game_status()
                if is_over:
                    if winner == 3:  # 平局
                        draws += 1
                        logging.info("平局")
                    elif winner == self.ai_color:
                        wins += 1
                        logging.info("AI获胜")
                    else:
                        losses += 1
                        logging.info("AI失败")
                else:
                    draws += 1
                    logging.info("平局")
        
        win_rate = wins / num_games * 100
        loss_rate = losses / num_games * 100
        draw_rate = draws / num_games * 100
        
        logging.info(f"测试结果: 胜率={win_rate:.2f}%, 败率={loss_rate:.2f}%, 平局率={draw_rate:.2f}%")
        return win_rate, loss_rate, draw_rate

    def check_win(self, board: np.ndarray, color: int) -> bool:
        """检查指定颜色的棋子是否连成5子
        参数：
        - board: 棋盘状态
        - color: 棋子颜色（1=黑子，2=白子）
        
        返回值：
        - True: 连成5子
        - False: 未连成5子
        """
        # 检查所有方向：水平、垂直、对角线
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        
        for dx, dy in directions:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    # 检查从(i,j)开始的连续5个位置
                    count = 0
                    for k in range(5):
                        x = i + k * dx
                        y = j + k * dy
                        # 检查是否在棋盘范围内
                        if 0 <= x < self.board_size and 0 <= y < self.board_size:
                            if board[x][y] == color:
                                count += 1
                            else:
                                break
                        else:
                            break
                    if count == 5:
                        return True
        return False

if __name__ == "__main__":
    # 训练模式
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        agent = QLearningAgent(ai_color=1)
        agent.train(num_episodes=1000)
        # 保存训练好的模型
        agent.save_model("gomoku_model.pth")
    
    # 测试模式
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        agent = QLearningAgent(ai_color=1, model_path="gomoku_model.pth")
        agent.test(num_games=100)