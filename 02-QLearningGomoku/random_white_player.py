import numpy as np
import requests
import logging
from datetime import datetime
import time
import random
from typing import Tuple, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'random_white_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class RandomWhitePlayer:
    def __init__(self, board_size: int = 8):
        self.board_size = board_size
        self.color = 1  # 白子
        self.base_url = "http://localhost:8080"
        logging.info(f"初始化随机白子玩家: 棋盘大小={board_size}")
    
    def get_board_state(self) -> np.ndarray:
        """获取当前棋盘状态"""
        try:
            response = requests.get(f"{self.base_url}/get_board")
            data = response.json()
            logging.debug(f"获取棋盘状态: 状态码={response.status_code}, 响应={data}")
            return np.array(data["board"])
        except Exception as e:
            logging.error(f"获取棋盘状态失败: {str(e)}")
            raise
    
    def can_move(self) -> Tuple[bool, int]:
        """检查是否可以移动，并返回当前玩家"""
        try:
            response = requests.get(f"{self.base_url}/can_move")
            data = response.json()
            logging.debug(f"检查移动状态: 状态码={response.status_code}, 响应={data}")
            return data["can_move"], data["current_player"]
        except Exception as e:
            logging.error(f"检查移动状态失败: {str(e)}")
            raise
    
    def make_move(self, x: int, y: int) -> bool:
        """在指定位置落子"""
        try:
            data = {
                "x": int(x),
                "y": int(y),
                "color": self.color
            }
            logging.info(f"尝试落子: 位置=({x}, {y})")
            response = requests.post(f"{self.base_url}/move", json=data)
            result = response.json()
            logging.info(f"落子结果: 状态码={response.status_code}, 响应={result}")
            return result["success"]
        except Exception as e:
            logging.error(f"落子失败: {str(e)}")
            raise
    
    def get_valid_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """获取所有合法的移动位置，返回 (x, y)"""
        valid_moves = []
        for y in range(self.board_size):       # 先行号 y
            for x in range(self.board_size):   # 后列号 x
                if board[y][x] == 0:           # 注意仍然用 [y][x] 访问
                    valid_moves.append((x, y)) # 追加顺序改为 (x, y)
        return valid_moves
    
    def select_random_move(self, board: np.ndarray) -> Tuple[int, int]:
        """随机选择一个合法的移动位置"""
        valid_moves = self.get_valid_moves(board)
        return random.choice(valid_moves)
    
    def play_game(self):
        """开始一局游戏"""
        logging.info("开始新游戏")
        while True:
            can_play, current = self.can_move()
            logging.info(f"检查游戏状态: can_play={can_play}, current_player={current}, 我是白子({self.color})")
            
            if not can_play:
                logging.info("游戏已结束")
                break
            
            if current == self.color:  # 只有当当前玩家是白子时才尝试落子
                logging.info("轮到白子落子")
                # 白子的回合
                board = self.get_board_state()  # 获取最新棋盘状态
                logging.info(f"当前棋盘状态:\n{board}")
                
                x, y = self.select_random_move(board)
                logging.info(f"随机选择落子位置: ({x}, {y})")
                
                success = self.make_move(x, y)
                logging.info(f"落子结果: success={success}")
                
                if not success:
                    logging.warning("落子失败，尝试其他位置")
                    time.sleep(0.1)  # 等待一小段时间后重试
                    continue
            else:
                # 等待黑子落子
                logging.info(f"等待黑子落子... (当前玩家={current})")
                time.sleep(0.1)

if __name__ == "__main__":
    player = RandomWhitePlayer()
    while True:
        try:
            player.play_game()
            logging.info("一局游戏结束，开始新的一局...")
            time.sleep(1)  # 等待一秒后开始新游戏
        except Exception as e:
            logging.error(f"游戏过程中发生错误: {str(e)}")
            time.sleep(5)  # 发生错误时等待5秒后重试 