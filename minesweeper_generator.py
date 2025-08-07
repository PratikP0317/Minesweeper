import numpy as np
import random
from typing import Tuple, List, Optional

class MinesweeperGenerator:
    """
    Minesweeper board generator that works like Google's Minesweeper.
    Generates boards with safe first click and creates two matrices:
    - Game board: 0-8 for revealed numbers, -1 for unrevealed
    - Mine board: Binary matrix (1 = mine, 0 = safe)
    """
    
    def __init__(self):
        # Google-like difficulty presets (rows, cols, mines)
        self.difficulties = {
            'beginner': (9, 9, 10),
            'intermediate': (16, 16, 40),
            'expert': (16, 30, 99)
        }
    
    def generate_board(self, difficulty: str = 'beginner', first_click: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Minesweeper board where the first click is guaranteed to be safe.
        
        Args:
            difficulty: 'beginner', 'intermediate', or 'expert'
            first_click: (row, col) of the first click. If None, will be randomly chosen.
        
        Returns:
            Tuple of (game_board, mine_board) where:
            - game_board: 2D array with 0-8 for revealed numbers, -1 for unrevealed
            - mine_board: 2D array with 1 for mines, 0 for safe cells
        """
        rows, cols, num_mines = self.difficulties[difficulty]
        
        # Initialize empty boards
        mine_board = np.zeros((rows, cols), dtype=int)
        game_board = np.full((rows, cols), -1, dtype=int)  # All cells start unrevealed
        
        # If no first click specified, choose randomly
        if first_click is None:
            first_click = (random.randint(0, rows-1), random.randint(0, cols-1))
        
        first_row, first_col = first_click
        
        # Generate safe zone around first click (3x3 area)
        safe_zone = self._get_safe_zone(first_row, first_col, rows, cols)
        
        # Place mines randomly, avoiding the safe zone
        mine_positions = self._place_mines(rows, cols, num_mines, safe_zone)
        
        # Update mine board
        for row, col in mine_positions:
            mine_board[row, col] = 1
        
        # Calculate number board (adjacent mine counts)
        number_board = self._calculate_numbers(mine_board)
        
        return game_board, mine_board, number_board
    
    def _get_safe_zone(self, row: int, col: int, rows: int, cols: int) -> List[Tuple[int, int]]:
        """Get all positions in the 3x3 safe zone around the first click."""
        safe_zone = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    safe_zone.append((new_row, new_col))
        return safe_zone
    
    def _place_mines(self, rows: int, cols: int, num_mines: int, safe_zone: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Place mines randomly, avoiding the safe zone."""
        all_positions = [(r, c) for r in range(rows) for c in range(cols)]
        available_positions = [pos for pos in all_positions if pos not in safe_zone]
        
        if len(available_positions) < num_mines:
            raise ValueError(f"Not enough space to place {num_mines} mines")
        
        mine_positions = random.sample(available_positions, num_mines)
        return mine_positions
    
    def _calculate_numbers(self, mine_board: np.ndarray) -> np.ndarray:
        """Calculate the number board based on adjacent mines."""
        rows, cols = mine_board.shape
        number_board = np.zeros((rows, cols), dtype=int)
        
        for row in range(rows):
            for col in range(cols):
                if mine_board[row, col] == 1:
                    number_board[row, col] = -1  # Mine (we'll handle this differently)
                else:
                    # Count adjacent mines
                    count = 0
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if dr == 0 and dc == 0:
                                continue
                            new_row, new_col = row + dr, col + dc
                            if (0 <= new_row < rows and 0 <= new_col < cols and 
                                mine_board[new_row, new_col] == 1):
                                count += 1
                    number_board[row, col] = count
        
        return number_board
    
    def click_cell(self, game_board: np.ndarray, number_board: np.ndarray, click_row: int, click_col: int) -> np.ndarray:
        """
        Simulate clicking a cell and revealing the board.
        
        Args:
            game_board: Current game board (-1 for unrevealed, 0-8 for revealed)
            number_board: Number board with adjacent mine counts
            click_row, click_col: Position to click
        
        Returns:
            Updated game board after the click
        """
        rows, cols = game_board.shape
        new_game_board = game_board.copy()
        
        def flood_fill(row: int, col: int):
            if (row < 0 or row >= rows or col < 0 or col >= cols or 
                new_game_board[row, col] != -1):
                return
            
            # Reveal the cell
            new_game_board[row, col] = number_board[row, col]
            
            # If it's empty (0), flood fill neighbors
            if number_board[row, col] == 0:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        flood_fill(row + dr, col + dc)
        
        flood_fill(click_row, click_col)
        return new_game_board

    def print_board(self, board: np.ndarray, title: str = "Board"):
        """Print a board in a readable format."""
        print(f"\n{title}:")
        rows, cols = board.shape
        print("  " + " ".join(f"{i:2}" for i in range(cols)))
        print("  " + "-" * (cols * 3))
        for i in range(rows):
            row_str = f"{i:2}|"
            for j in range(cols):
                val = board[i, j]
                if val == -1:
                    row_str += " ."  # Unrevealed
                else:
                    row_str += f" {val}"
            print(row_str)


# Example usage and testing
if __name__ == "__main__":
    generator = MinesweeperGenerator()
    
    # Generate a single board
    print("Generating a beginner board...")
    game_board, mine_board, number_board = generator.generate_board('beginner', first_click=(4, 4))
    
    generator.print_board(mine_board, "Mine Board (1=mine, 0=safe)")
    generator.print_board(number_board, "Number Board (0-8=adjacent mines)")
    generator.print_board(game_board, "Game Board (-1=unrevealed)")
    
    # Show what happens after first click
    revealed_board = generator.click_cell(game_board, number_board, 4, 4)
    generator.print_board(revealed_board, "Game Board after first click (4,4)")
    
    # Generate training data
    print("\nGenerating training data...")
    game_boards, mine_boards = generator.generate_training_data(5, 'beginner')
    print(f"Generated {len(game_boards)} boards with shape {game_boards[0].shape}")
    
    # Create ML dataset
    print("\nCreating ML dataset...")
    X, y = generator.create_ml_dataset(10, 'beginner', first_click=(4, 4))
    print(f"ML dataset shapes: X={X.shape}, y={y.shape}")
    
    # Test different difficulties
    for difficulty in ['beginner', 'intermediate', 'expert']:
        rows, cols, mines = generator.difficulties[difficulty]
        print(f"\n{difficulty.capitalize()}: {rows}x{cols} with {mines} mines")
        game_board, mine_board, number_board = generator.generate_board(difficulty)
        print(f"Board shape: {game_board.shape}") 