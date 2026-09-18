import pygame
from pygame.locals import *
import logging
from config.basic import ROW_NUMBER, COL_NUMBER, LOGS_DIR
from config.render import BACKGROUND_COLOR, BLOCK_COLOR
from board import Board

logger = logging.getLogger("game.render")
# tmp log output
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

blockRects = []


def start(board:Board):
    pygame.init()
    screen: pygame.Surface = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    pygame.display.set_caption("DigitalHuarongPass")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == MOUSEBUTTONDOWN:
                # logger.info(f"Mouse button pressed at position {event.pos}")
                mouse_pos = event.pos
                block_num = (-1, -1)
                for i, rects in enumerate(blockRects):
                    for j, rect in enumerate(rects):
                        if rect.collidepoint(mouse_pos):
                            logger.info(f"Block {i},{j} clicked at position {mouse_pos}")
                            block_num = (i, j)
                            break
                    if block_num != (-1, -1):
                        break

        screen.fill(BACKGROUND_COLOR)  # Fill the screen with the background color
        # Draw game elements here
        block_size = min(
            (
                screen.get_width() // (COL_NUMBER + 2),
                screen.get_height() // (ROW_NUMBER + 2),
            )
        )
        start_x = (screen.get_width() - (block_size * (COL_NUMBER))) // 2
        start_y = (screen.get_height() - (block_size * (ROW_NUMBER))) // 2
        for row in range(ROW_NUMBER + 1):
            pygame.draw.line(
                screen,
                BLOCK_COLOR,
                (start_x, start_y + row * block_size),
                (start_x + block_size * COL_NUMBER, start_y + row * block_size),
            )
        for col in range(COL_NUMBER + 1):
            pygame.draw.line(
                screen,
                BLOCK_COLOR,
                (start_x + col * block_size, start_y),
                (start_x + col * block_size, start_y + block_size * ROW_NUMBER),
            )
        blockRects.clear()
        for row in range(ROW_NUMBER):
            tmp = []
            for col in range(COL_NUMBER):
                rect = pygame.Rect(
                    start_x + col * block_size,
                    start_y + row * block_size,
                    block_size,
                    block_size,
                )
                tmp.append(rect)
                text = str(board.board[row][col]) if board.board[row][col] != -1 else ""
                font = pygame.font.Font(None, int(block_size * 0.5))
                text_surface = font.render(text, True, BLOCK_COLOR)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)

            blockRects.append(tmp)

        pygame.display.flip()  # Update the display

    pygame.quit()


def main() -> None:
    """控制台入口：建立棋盘、初始化日志并启动游戏窗口。"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger_ = logging.getLogger("game")
    logger_.setLevel(logging.DEBUG)
    logger_.addHandler(logging.StreamHandler())
    logger_.addHandler(logging.FileHandler(LOGS_DIR / "game.log", mode="w"))

    start(Board(COL_NUMBER, ROW_NUMBER))


if __name__ == "__main__":
    main()
