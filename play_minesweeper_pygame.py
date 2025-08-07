#!/usr/bin/env python3
"""
Minesweeper game as a gym environment for RL training.
"""

import pygame
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minesweeper_generator import MinesweeperGenerator

class MinesweeperGame(gym.Env):
    def __init__(self, difficulty='beginner', render=True):
        self.generator = MinesweeperGenerator()
        self.difficulty = difficulty
        self.rows, self.cols, self.mines = self.generator.difficulties[difficulty]
        
        # Generate the board
        self.game_board, self.mine_board, self.number_board = self.generator.generate_board(difficulty)
        self.revealed_board = self.game_board.copy()
        self.game_over = False
        self.won = False
        self.first_click = True
        
        # Rendering setup
        self.render = render
        if self.render:
            # Pygame setup
            self.cell_size = 30
            self.margin = 50
            self.width = self.cols * self.cell_size + 2 * self.margin
            self.height = self.rows * self.cell_size + 2 * self.margin + 100  # Extra space for info
            
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(f"Minesweeper - {difficulty.capitalize()}")
            
            # Colors
            self.WHITE = (255, 255, 255)
            self.GRAY = (192, 192, 192)
            self.DARK_GRAY = (128, 128, 128)
            self.BLACK = (0, 0, 0)
            self.RED = (255, 0, 0)
            self.BLUE = (0, 0, 255)
            self.GREEN = (0, 128, 0)
            self.PURPLE = (128, 0, 128)
            self.MAROON = (128, 0, 0)
            self.TURQUOISE = (0, 128, 128)
            self.ORANGE = (255, 165, 0)
            
            # Font
            self.font = pygame.font.Font(None, 20)
            self.large_font = pygame.font.Font(None, 36)
        
    def get_cell_color(self, val):
        """Get color for number values."""
        colors = {
            1: self.BLUE, 2: self.GREEN, 3: self.RED, 4: self.PURPLE,
            5: self.MAROON, 6: self.TURQUOISE, 7: self.BLACK, 8: self.GRAY
        }
        return colors.get(val, self.BLACK)
    
    def draw_board(self):
        """Draw the game board."""
        if not self.render:
            return
            
        self.screen.fill(self.WHITE)
        
        # Draw cells
        for row in range(self.rows):
            for col in range(self.cols):
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                
                val = self.revealed_board[row, col]
                
                # Draw cell background
                if val == -1:  # Unrevealed
                    pygame.draw.rect(self.screen, self.GRAY, (x, y, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, self.cell_size, self.cell_size), 2)
                elif val == -2:  # Flagged
                    pygame.draw.rect(self.screen, self.GRAY, (x, y, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, self.cell_size, self.cell_size), 2)
                    # Draw flag (simple red square)
                    pygame.draw.rect(self.screen, self.RED, (x + 5, y + 5, self.cell_size - 10, self.cell_size - 10))
                elif val == -3:  # Mine (game over)
                    pygame.draw.rect(self.screen, self.RED, (x, y, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, self.cell_size, self.cell_size), 2)
                else:  # Revealed number
                    pygame.draw.rect(self.screen, self.WHITE, (x, y, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, self.cell_size, self.cell_size), 2)
                    
                    if val > 0:
                        # Draw number
                        text = self.font.render(str(val), True, self.get_cell_color(val))
                        text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                        self.screen.blit(text, text_rect)
        
        # Draw grid lines
        for i in range(self.rows + 1):
            y = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.DARK_GRAY, (self.margin, y), (self.width - self.margin, y))
        
        for i in range(self.cols + 1):
            x = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.DARK_GRAY, (x, self.margin), (x, self.height - 100))
        
        # Draw info
        info_y = self.height - 80
        mines_text = f"Mines: {self.mines}"
        revealed_text = f"Revealed: {np.sum(self.revealed_board >= 0)}"
        
        mines_surface = self.font.render(mines_text, True, self.BLACK)
        revealed_surface = self.font.render(revealed_text, True, self.BLACK)
        
        self.screen.blit(mines_surface, (self.margin, info_y))
        self.screen.blit(revealed_surface, (self.margin + 150, info_y))
        
        # Draw game status
        if self.game_over:
            if self.won:
                status_text = "🎉 You Won!"
                status_color = self.GREEN
            else:
                status_text = "💥 Game Over!"
                status_color = self.RED
        else:
            status_text = "Playing..."
            status_color = self.BLACK
        
        status_surface = self.large_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(center=(self.width // 2, info_y + 30))
        self.screen.blit(status_surface, status_rect)
        
        pygame.display.flip()
    
    def get_cell_from_pos(self, pos):
        """Convert mouse position to cell coordinates."""
        if not self.render:
            return None
        x, y = pos
        if (self.margin <= x < self.width - self.margin and 
            self.margin <= y < self.height - 100):
            col = (x - self.margin) // self.cell_size
            row = (y - self.margin) // self.cell_size
            if 0 <= row < self.rows and 0 <= col < self.cols:
                return row, col
        return None
    
    def click_cell(self, row, col):
        """Click a cell and reveal it."""
        if self.game_over or row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        
        if self.revealed_board[row, col] != -1:
            return False
        
        # First click is always safe
        if self.first_click:
            # Regenerate board with this click as safe
            self.game_board, self.mine_board, self.number_board = self.generator.generate_board(
                self.difficulty, first_click=(row, col)
            )
            self.first_click = False
        
        # Check if clicked on mine
        if self.mine_board[row, col] == 1:
            self.game_over = True
            self.reveal_all_mines()
            return False
        
        # Reveal the cell
        self.revealed_board = self.generator.click_cell(self.revealed_board, self.number_board, row, col)
        
        # Check if won
        if self.check_win():
            self.game_over = True
            self.won = True
        return True
    
    def flag_cell(self, row, col):
        """Flag/unflag a cell."""
        if self.game_over or row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        
        if self.revealed_board[row, col] == -1:
            self.revealed_board[row, col] = -2  # Flag
        elif self.revealed_board[row, col] == -2:
            self.revealed_board[row, col] = -1  # Unflag
        
        return True
    
    def reveal_all_mines(self):
        """Reveal all mines when game is over."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.mine_board[i, j] == 1:
                    self.revealed_board[i, j] = -3  # Show mine
    
    def check_win(self):
        """Check if the game is won."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.mine_board[i, j] == 0 and self.revealed_board[i, j] == -1:
                    return False
        return True
    
    
    def get_state(self):
        return self.revealed_board.copy()
    
    def reset(self, seed=None):
        """Reset the game to initial state (gym.Env method)."""
        super().reset(seed=seed)
        self.game_board, self.mine_board, self.number_board = self.generator.generate_board(self.difficulty)
        self.revealed_board = self.game_board.copy()
        self.game_over = False
        self.won = False
        self.first_click = True
        self.game_over = False
        return self.get_state()
    
    def step(self, action):
        """Take a step in the environment (gym.Env method)."""
        # Convert action to (row, col)
        row, col = action
        
        # Make the move
        self.click_cell(row, col)
        
        # Get current state
        state = self.get_state()
        
        # Calculate reward
        if self.game_over:
            if self.won:
                reward = 1
            else:
                reward = -1
        else:
            reward = .02
        
        # Check if done
        done = self.game_over
        
        # Info dict
        info = {
            'won': self.won,
            'cells_revealed': np.sum(self.revealed_board >= 0),
            'mines_remaining': self.mines - np.sum(self.revealed_board == -2)
        }
        
        return reward, done, info
    
    def _get_obs(self):
        """Get the current observation (revealed board)."""
        return self.revealed_board.copy()
    
    def run(self):
        """Main game loop."""
        if not self.render:
            # Non-rendering mode - just return after initialization
            return
            
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    cell = self.get_cell_from_pos(event.pos)
                    if cell:
                        row, col = cell
                        if event.button == 1:  # Left click
                            self.click_cell(row, col)
                        elif event.button == 3:  # Right click
                            self.flag_cell(row, col)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Restart
                        self.__init__(self.difficulty, render=self.render)
                    elif event.key == pygame.K_q:  # Quit
                        running = False
            
            self.draw_board()
        
        pygame.quit()
        sys.exit()

def main():
    print("=== Minesweeper Game (Pygame) ===")
    print("Choose difficulty:")
    print("1. Beginner (9x9, 10 mines)")
    print("2. Intermediate (16x16, 40 mines)")
    print("3. Expert (16x30, 99 mines)")
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == '1':
            difficulty = 'beginner'
            break
        elif choice == '2':
            difficulty = 'intermediate'
            break
        elif choice == '3':
            difficulty = 'expert'
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\nControls:")
    print("- Left click: Reveal cell")
    print("- Right click: Flag/unflag cell")
    print("- R: Restart game")
    print("- Q: Quit game")
    print("\nStarting game...")
    
    # Create and start the game
    game = MinesweeperGame(difficulty)
    game.run()

if __name__ == "__main__":
    main() 