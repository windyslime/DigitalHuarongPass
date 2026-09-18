import random

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

    if not solvable:
        positions = [(r, c) for r in range(row) for c in range(col) if board[r][c] != -1]
        (r1, c1), (r2, c2) = random.sample(positions, 2)
        board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]

    return board
