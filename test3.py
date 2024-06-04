import numpy as np
import matplotlib.pyplot as plt

# 设置参数
L = 100  # 网格大小
iterations = 100000  # 迭代次数
K = 0.1  # 决策噪音
ce = 10   # 排除成本
r = 4  # 公共品游戏的收益比

# 初始化策略网格：0 - c, 1 - EC, 2 - ED, 3 - D
strategy_grid = np.random.choice([0, 1, 2, 3], size=(L, L), p=[0.4, 0.1, 0, 0.5])
exclusion_next = np.zeros((L, L), dtype=bool)  # 排除标记

def find_neighbors(x, y):
    """ 返回邻居的坐标 """
    return [(x, (y+1) % L), (x, (y-1) % L), ((x+1) % L, y), ((x-1) % L, y)]

def play_game(center_x, center_y, target_x, target_y):
    """ 执行一轮公共品博弈，计算目标节点的收益 """
    if exclusion_next[target_x, target_y]:
        return -ce if strategy_grid[target_x, target_y] == 2 else 0

    num_cooperators = 0
    num_participants = 0
    neighbors = find_neighbors(center_x, center_y)

    for nx, ny in neighbors + [(center_x, center_y)]:
        if not exclusion_next[nx, ny]:  # 只计算未被排除的个体
            num_participants += 1
            if strategy_grid[nx, ny] in [0, 1]:  # 合作者
                num_cooperators += 1

    if strategy_grid[target_x, target_y] in [0, 1]:  # 合作者
        payoff = (r * num_cooperators / num_participants) - 1
    elif strategy_grid[target_x, target_y] == 3:  # 背叛者
        payoff = r * num_cooperators / num_participants
    elif strategy_grid[target_x, target_y] == 2:  # 驱逐者
        payoff = r * num_cooperators / num_participants - ce

    return payoff

def exclusion_phase(x, y):
    """ 执行排除阶段 """
    participants = [strategy_grid[x, y]]
    neighbors = find_neighbors(x, y)
    for nx, ny in neighbors:
        participants.append(strategy_grid[nx, ny])

    if any(s in [1, 2] for s in participants):  # 如果含有驱逐合作者或驱逐背叛者
        if strategy_grid[x, y] == 3:  # 排除中心节点的背叛者
            exclusion_next[x, y] = True
        for nx, ny in neighbors:
            if strategy_grid[nx, ny] == 3:  # 排除邻居中的背叛者
                exclusion_next[nx, ny] = True

    if participants.count(2) > 1:  # 如果含有两个及以上驱逐背叛者
        if strategy_grid[x, y] == 2:  # 排除中心节点的驱逐背叛者
            exclusion_next[x, y] = True
        for nx, ny in neighbors:
            if strategy_grid[nx, ny] == 2:  # 排除邻居中的驱逐背叛者
                exclusion_next[nx, ny] = True

def calculate_total_payoff(x, y):
    """ 计算五轮总收益 """
    total_payoff = 0
    exclusion_phase(x, y)
    # 以自身为中心计算收益
    total_payoff += play_game(x, y, x, y)

    # 以邻居为中心计算收益
    for nx, ny in find_neighbors(x, y):
        exclusion_phase(nx, ny)
        total_payoff += play_game(nx, ny, x, y)

    return total_payoff

def update_strategy(x, y):
    """ 根据费米规则更新策略 """
    current_payoff = calculate_total_payoff(x, y)
    nx, ny = find_neighbors(x, y)[np.random.randint(4)]  # 随机选择一个邻居
    neighbor_payoff = calculate_total_payoff(nx, ny)
    if np.random.rand() < 1 / (1 + np.exp((current_payoff - neighbor_payoff) / K)):
        strategy_grid[x, y] = strategy_grid[nx, ny]

# 主循环
for iteration in range(iterations):
    exclusion_next.fill(False)
    for _ in range(L * L):
        x, y = np.random.randint(L), np.random.randint(L)

        update_strategy(x, y)  # 策略更新阶段
    # 每一定次数后输出当前合作者比例
    if (iteration + 1) % 100 == 0:
        num_cooperators = np.sum(strategy_grid == 0) + np.sum(strategy_grid == 1)
        print(f"Iteration {iteration + 1}: Cooperator proportion = {num_cooperators / (L * L):.4f}")

# 结束后可视化策略分布
plt.imshow(strategy_grid, cmap='nipy_spectral', origin='upper')
plt.colorbar(label='Strategy')
plt.title('Strategy distribution after simulation')
plt.show()
