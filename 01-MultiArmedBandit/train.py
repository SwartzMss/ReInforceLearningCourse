import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用支持中文的字体，如黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from current font", module="tkinter")

class EpsilonGreedyBandit:
    def __init__(self, true_means, epsilon):
        """
        初始化多臂老虎机环境与算法参数

        参数:
        - true_means: 每个老虎机的真实奖励均值（用于模拟真实环境）
        - epsilon: 探索概率（例如 0.1 表示 10% 的概率随机探索）
        """
        self.true_means = true_means        # 环境中各个老虎机的真实均值（实际中未知）
        self.epsilon = epsilon              # 探索率
        self.n_arms = len(true_means)         # 老虎机数量
        self.estimates = np.zeros(self.n_arms)  # 初始化每个臂的奖励均值估计
        self.counts = np.zeros(self.n_arms)     # 记录每个臂被选择的次数

    def select_arm(self):
        """
        根据 ε-贪婪策略选择一个老虎机

        返回:
        - 选择的老虎机的编号
        """
        if np.random.rand() < self.epsilon:
            # 探索：随机选择一个老虎机
            return np.random.randint(self.n_arms)
        else:
            # 利用：选择当前估计奖励最高的老虎机
            return np.argmax(self.estimates)

    def update(self, arm, reward):
        """
        更新被选择老虎机的奖励估计

        参数:
        - arm: 被选择的老虎机编号
        - reward: 获得的奖励
        """
        self.counts[arm] += 1
        # 使用增量更新公式：new_estimate = old_estimate + (reward - old_estimate) / count
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

def train_bandit(true_means, epsilon, iterations):
    """
    训练阶段：利用 ε-贪婪策略训练多臂老虎机

    参数:
    - true_means: 两个老虎机的真实均值
    - epsilon: 探索率
    - iterations: 训练过程中进行的总尝试次数

    返回:
    - bandit: 训练后的老虎机对象（包含学习到的奖励估计）
    - training_rewards: 每次训练获得的奖励列表
    """
    bandit = EpsilonGreedyBandit(true_means, epsilon)
    training_rewards = []  # 记录训练过程中获得的奖励

    for i in range(iterations):
        arm = bandit.select_arm()          # 根据当前策略选择一个老虎机
        # 模拟获得奖励：假设奖励服从均值为 true_means[arm]，标准差为1的正态分布
        reward = np.random.randn() + true_means[arm]
        bandit.update(arm, reward)         # 更新该老虎机的估计
        training_rewards.append(reward)

    return bandit, training_rewards

def test_bandit(bandit, test_iterations):
    """
    测试阶段：使用训练后学习到的最佳策略进行测试

    参数:
    - bandit: 训练后的老虎机对象
    - test_iterations: 测试过程中进行的尝试次数

    返回:
    - avg_test_reward: 测试阶段的平均奖励
    """
    test_rewards = []
    # 测试时我们固定选择训练过程中估计最优的老虎机（贪婪策略）
    best_arm = np.argmax(bandit.estimates)
    print(f"测试时选择的最佳老虎机为：{best_arm}，其估计奖励为：{bandit.estimates[best_arm]:.4f}")
    
    for i in range(test_iterations):
        reward = np.random.randn() + bandit.true_means[best_arm]
        test_rewards.append(reward)

    avg_test_reward = np.mean(test_rewards)
    return avg_test_reward

# 设置随机种子，保证结果可重复
np.random.seed(42)

# 定义两个老虎机的真实均值，例如：老虎机0的均值为1.0，老虎机1的均值为1.8
true_means = [1.0, 1.8]
epsilon = 0.1            # 训练阶段采用10%的探索率
training_iterations = 1000  # 训练过程中进行1000次尝试
test_iterations = 500       # 测试过程中进行500次尝试

# --------------------
# 训练阶段
# --------------------
bandit, training_rewards = train_bandit(true_means, epsilon, training_iterations)

# 显示训练后每个老虎机的奖励估计和选择次数
print("训练结束后的奖励估计：", bandit.estimates)
print("各老虎机被选择的次数：", bandit.counts)

# 绘制训练过程中累计平均奖励的变化情况
training_cum_avg = np.cumsum(training_rewards) / (np.arange(training_iterations) + 1)
plt.figure(figsize=(10,5))
plt.plot(training_cum_avg)
plt.xlabel("训练迭代次数")
plt.ylabel("累计平均奖励")
plt.title("训练阶段表现")
plt.show()

# --------------------
# 测试阶段
# --------------------
avg_test_reward = test_bandit(bandit, test_iterations)
print(f"测试阶段的平均奖励：{avg_test_reward:.4f}")
