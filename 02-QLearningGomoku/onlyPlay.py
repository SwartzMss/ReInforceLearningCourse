import numpy as np
import requests
import torch
import torch.nn as nn
import logging
from datetime import datetime
import os
import sys
import time
import random
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gomoku_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class QLearningAgent:
    def __init__(self, board_size: int = 8, ai_color: int = 2, model_path: str = None):
        self.board_size = board_size
        self.ai_color = ai_color
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 0.0  # 完全贪婪
        input_size = board_size * board_size * 2
        output_size = board_size * board_size
        self.q_network = QNetwork(input_size, output_size).to(self.device)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logging.info(f"已加载模型: {model_path}")

        self.base_url = "http://localhost:8080"

    def board_to_state(self, board: np.ndarray) -> torch.Tensor:
        black = (board == self.ai_color).astype(np.float32)
        white = (board == 3 - self.ai_color).astype(np.float32)
        state = np.stack([black, white])
        return torch.FloatTensor(state.flatten()).to(self.device)

    def select_action(self, state: torch.Tensor) -> Tuple[int, int]:
        board = state.view(2, self.board_size, self.board_size).sum(dim=0).detach().cpu().numpy()
        valid_moves = [(x, y) for y in range(self.board_size) for x in range(self.board_size) if board[y][x] == 0]
        if random.random() < self.epsilon or not valid_moves:
            return random.choice(valid_moves)
        with torch.no_grad():
            q_values = self.q_network(state).detach().cpu().numpy()
            for y in range(self.board_size):
                for x in range(self.board_size):
                    idx = y * self.board_size + x
                    if (x, y) not in valid_moves:
                        q_values[idx] = -np.inf
            best_idx = np.argmax(q_values)
            return (best_idx % self.board_size, best_idx // self.board_size)

    def can_move(self) -> Tuple[bool, int]:
        response = requests.get(f"{self.base_url}/can_move")
        data = response.json()
        return data["can_move"], data["current_player"]

    def get_board_state(self) -> np.ndarray:
        response = requests.get(f"{self.base_url}/get_board")
        return np.array(response.json()["board"])

    def make_move(self, x: int, y: int, color: int) -> Tuple[bool, bool, int]:
        response = requests.post(
            f"{self.base_url}/move",
            json={"x": int(x), "y": int(y), "color": int(color)}
        )
        result = response.json()
        response = requests.get(f"{self.base_url}/get_board")
        data = response.json()
        is_over = data["winner"] != 0 or np.all(np.array(data["board"]) != 0)
        return result["success"], is_over, data["winner"]


    def reset_game(self):
        requests.post(f"{self.base_url}/reset")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])

if __name__ == "__main__":
    model_path = "gomoku_model_train.pth"
    agent = QLearningAgent(ai_color=1, model_path=model_path if os.path.exists(model_path) else None)
    logging.info("进入 only-play 陪练模式（不训练，仅自动下棋）")

    while True:
        try:
            can_play, current = agent.can_move()
            if not can_play:
                time.sleep(0.1)
                continue

            if current == agent.ai_color:
                board = agent.get_board_state()
                state = agent.board_to_state(board)
                action = agent.select_action(state)
                success, is_over, _ = agent.make_move(action[0], action[1], agent.ai_color)
                if not success:
                    logging.warning(f"落子失败: {action}")
                # 👇不要 reset，让训练 AI 控制 reset
            else:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("陪练进程终止")
            break


