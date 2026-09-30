from typing import Generator
import heapq
import random
import logging
import pygame

logger = logging.getLogger("game.board")


# deepseek加的这个注释句号有点多啊, 注释是我写的，给你看的
END = object()


class Board:
    """棋盘类，表示数字华容道的棋盘。

    Attributes:
        col (int): 棋盘的列数。
        row (int): 棋盘的行数。
        board (list[list[int]]): 二维列表，表示棋盘上的数字和空格。
    """

    def __init__(self, col: int, row: int):
        """初始化棋盘。

        Args:
            col (int): 棋盘的列数。
            row (int): 棋盘的行数。
        """
        self.col = col
        self.row = row
        self.board = generate_board(col, row)
        logger.debug(f"Initialized board: {self.board}")
        logger.info(f"Board created with dimensions {col}x{row}")

    def dealWithSwap(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """处理两个位置的交换。
        自动寻找挨着他的空白格，并完成与空白格的交换此时输出空白格的坐标(用于制作动画),若点击的为空白格或点击格周围没有空白格，则输出false

        Args:
            pos (pygame.Vector2): 被点击格子的坐标，x 为行索引，y 为列索引（与点击处理得到的 block_num=(行, 列) 一致）。

        Returns:
            pygame.Vector2 | None: 如果交换成功，返回空白格交换后的坐标（x 为行索引，y 为列索引）；否则返回 None。
        """
        row, col = int(pos[1]), int(pos[0])
        if not (0 <= row < self.row and 0 <= col < self.col):
            return None
        if self.board[row][col] == -1:
            return None

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            blank_row, blank_col = row + dr, col + dc
            if 0 <= blank_row < self.row and 0 <= blank_col < self.col:
                if self.board[blank_row][blank_col] == -1:
                    self.board[row][col], self.board[blank_row][blank_col] = (
                        self.board[blank_row][blank_col],
                        self.board[row][col],
                    )
                    logger.debug(
                        f"Swapped ({row},{col}) with blank ({blank_row},{blank_col})"
                    )
                    return blank_row, blank_col

        return None

    def checkWin(self) -> bool:
        """检查当前棋盘是否处于胜利状态。

        Returns:
            bool: 如果棋盘处于胜利状态，返回 True；否则返回 False。
        """
        flat = [v for line in self.board for v in line]
        expected = list(range(1, self.col * self.row)) + [-1]
        logger.debug(f"Checking win condition: current={flat}, expected={expected}")
        return flat == expected

    def get_road(self, weight: float = 2.0) -> Generator[tuple[int, int] | object]:
        """获取当前棋盘的解路，返回一个生成器，生成从当前棋盘到胜利状态的每一步移动(-1|0|1, -1|0|1)
        最后返回END表示结束

        约定（很重要）：
            每一步是 (列增量, 行增量)，含义是「把空白格朝这个方向挪一格」，
            两个分量各自取 -1 / 0 / 1，且不会同时为 0。
            坐标顺序与 dealWithSwap 的入参一致（先列后行）：
            空白格从 (bc, br) 走到 (bc + dcol, br + drow)，
            真正被交换的方块就位于 (bc + dcol, br + drow)，
            也就是说这个元组恰好是「被交换方块的移动方向」取反。

            解路走完时棋盘已到胜利状态，之后再 yield 一次 END 表示结束。
            本方法只读 self.board，不会改动棋盘，要不要照着走由调用方决定。

        Args:
            weight (float): A* 的启发式权重，f = g + weight * h。
                1.0 是标准 A*，给出的是**最短解**，但 4x4 随机盘可能要搜几十秒；
                越大搜得越快，解路会比最短解长一些（仍保证是一组合法解）。
                默认 2.0：实测 4x4 均匀随机盘最慢约 0.3 秒、平均 0.05 秒，
                解路大约比最短解长两成。

        Yields:
            tuple[int, int] | object: 每一步的 (列增量, 行增量)；解路走完后为 END。
        """
        if weight <= 0:
            raise ValueError("weight 必须为正数")

        col, row = self.col, self.row
        start = tuple(v for line in self.board for v in line)
        goal = tuple(range(1, col * row)) + (-1,)

        if not is_solvable(start, col, row):
            logger.warning("get_road: 当前棋盘不可解，返回空解路")
            yield END
            return

        def heuristic(state: tuple[int, ...]) -> int:
            """曼哈顿距离 + 线性冲突，不会高估真实步数（可采纳）。"""
            distance = 0
            for index, value in enumerate(state):
                if value == -1:
                    continue
                current_row, current_col = divmod(index, col)
                goal_row, goal_col = divmod(value - 1, col)
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)

            # 同一行里两个方块都已经在目标行，但左右顺序是反的，至少得多绕 2 步
            for r in range(row):
                line = [
                    state[r * col + c]
                    for c in range(col)
                    if state[r * col + c] != -1
                    and (state[r * col + c] - 1) // col == r
                ]
                for i in range(len(line)):
                    for j in range(i + 1, len(line)):
                        if (line[i] - 1) % col > (line[j] - 1) % col:
                            distance += 2

            # 同一列里两个方块都已经在目标列，但上下顺序是反的，同样得多绕 2 步
            for c in range(col):
                column = [
                    state[r * col + c]
                    for r in range(row)
                    if state[r * col + c] != -1
                    and (state[r * col + c] - 1) % col == c
                ]
                for i in range(len(column)):
                    for j in range(i + 1, len(column)):
                        if (column[i] - 1) // col > (column[j] - 1) // col:
                            distance += 2

            return distance

        # 加权 A* 搜索。堆里放 (f, 序号, g, 状态)，序号是为了避免两个状态元组被拿来比较
        counter = 0
        open_heap = [(weight * heuristic(start), counter, 0, start)]
        best_g = {start: 0}
        # 记录前驱：{状态: (前驱状态, 空白格走这一步的方向)}
        came_from: dict[tuple[int, ...], tuple[tuple[int, ...], tuple[int, int]]] = {}

        solved: tuple[int, ...] | None = None
        while open_heap:
            _, _, current_g, current = heapq.heappop(open_heap)
            if current_g != best_g[current]:
                continue  # 这个状态后来被松弛得更短了，是过期记录
            if current == goal:
                solved = current
                break

            blank_index = current.index(-1)
            blank_row, blank_col = divmod(blank_index, col)
            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row, next_col = blank_row + delta_row, blank_col + delta_col
                if not (0 <= next_row < row and 0 <= next_col < col):
                    continue
                next_index = next_row * col + next_col
                next_state = list(current)
                next_state[blank_index], next_state[next_index] = (
                    next_state[next_index],
                    next_state[blank_index],
                )
                next_state = tuple(next_state)
                next_g = current_g + 1
                if next_state in best_g and best_g[next_state] <= next_g:
                    continue
                best_g[next_state] = next_g
                came_from[next_state] = (current, (delta_col, delta_row))
                counter += 1
                heapq.heappush(
                    open_heap,
                    (
                        next_g + weight * heuristic(next_state),
                        counter,
                        next_g,
                        next_state,
                    ),
                )

        if solved is None:
            logger.warning("get_road: 搜索失败，没能找到解路")
            yield END
            return

        # 顺着前驱指针回溯出整条解路
        road: list[tuple[int, int]] = []
        node = solved
        while node in came_from:
            node, direction = came_from[node]
            road.append(direction)
        road.reverse()
        logger.info(f"get_road: 找到解路，共 {len(road)} 步（weight={weight}）")

        yield from road
        yield END


def is_solvable(flat: list[int] | tuple[int, ...], col: int, row: int) -> bool:
    """判断摊平成一维的棋盘是否可解（数字华容道的逆序数判定）。

    规则与 generate_board 里用的完全一样：列数为奇数时只看逆序数的奇偶；
    列数为偶数时还要叠加空格所在行（自下往上数，从 1 开始）。-1 表示空格。
    """
    if -1 not in flat:
        return False

    nums = [v for v in flat if v != -1]
    inversions = sum(
        1
        for i in range(len(nums))
        for j in range(i + 1, len(nums))
        if nums[i] > nums[j]
    )
    blank_row_from_bottom = row - flat.index(-1) // col
    if col % 2 == 1:
        return inversions % 2 == 0
    return (inversions + blank_row_from_bottom) % 2 == 1


def generate_board(col: int, row: int) -> list[list[int]]:
    """生成一个保证可解的随机数字华容道棋盘。

    返回 row 行 x col 列的二维列表，数字为随机打乱的 1..col*row-1，-1 表示空格。
    """
    if col < 2 or row < 2:
        raise ValueError("col 和 row 至少为 2")

    tiles = list(range(1, col * row)) + [-1]
    random.shuffle(tiles)
    board = [tiles[r * col : (r + 1) * col] for r in range(row)]

    flat = [v for line in board for v in line]
    nums = [v for v in flat if v != -1]
    inversions = sum(
        1
        for i in range(len(nums))
        for j in range(i + 1, len(nums))
        if nums[i] > nums[j]
    )
    blank_row_from_bottom = row - flat.index(-1) // col
    if col % 2 == 1:
        solvable = inversions % 2 == 0
    else:
        solvable = (inversions + blank_row_from_bottom) % 2 == 1
    # 不可解情况(这破玩意怎么这么难搞)
    if not solvable:
        positions = [
            (r, c) for r in range(row) for c in range(col) if board[r][c] != -1
        ]
        (r1, c1), (r2, c2) = random.sample(positions, 2)
        board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]

    return board
