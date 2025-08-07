
import numpy as np

def get_neighbors(row, col, game):
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    open_coords = []
    flag_coords = []
    for dx, dy in neighbors:
        n_row, n_col = row + dx, col + dy
        if 0 <= n_row < game.rows and 0 <= n_col < game.cols:
            val = game.revealed_board[n_row, n_col]
            if val == -1:
                open_coords.append((n_row, n_col))
            elif val == -2:
                flag_coords.append((n_row, n_col))
    return open_coords, flag_coords

def simple_flag_strategy(game):
    revealed_board = game.get_state()
    unsolved_cells = game.get_unsolved_cells()
    found = False
    if len(unsolved_cells) == 0:
        return False
    for cell in unsolved_cells:
        row, col = cell
        value = revealed_board[row, col]
        open_cells, flag_cells = get_neighbors(row, col, game)
        # If the number of flags + open cells equals the cell value, flag all open cells
        if len(flag_cells) + len(open_cells) == value and len(open_cells) > 0:
            for open_cell in open_cells:
                game.flag_cell(open_cell[0], open_cell[1])
                print(f"flagged {open_cell} from {cell}")
                found = True
    return found

def simple_click_strategy(game):
    revealed_board = game.get_state()
    unsolved_cells = game.get_unsolved_cells()
    found = False
    if len(unsolved_cells) == 0:
        print("no unsolved cells")
        return False
    for cell in unsolved_cells:
        row, col = cell
        value = revealed_board[row, col]
        open_cells, flag_cells = get_neighbors(row, col, game)
        if len(flag_cells) == value and len(open_cells) > 0:
            for open_cell in open_cells:
                game.click_cell(open_cell[0], open_cell[1])
                print(f"clicked {open_cell} from {cell}")
                found = True
    return found

def click_most_likely_cell(game):
    revealed_board = game.get_state()
    unsolved_cells = game.get_unsolved_cells()
    if len(unsolved_cells) == 0:
        return False
    propability_board = np.zeros((game.rows, game.cols))
    for cell in unsolved_cells:
        row, col = cell
        value = revealed_board[row, col]
        open_cells, flag_cells = get_neighbors(row, col, game)
        value = value - len(flag_cells)
        value = value / len(open_cells)
        for open_cell in open_cells:
            propability_board[open_cell[0], open_cell[1]] += value
    max_value = np.max(propability_board)
    for row in range(game.rows):
        for col in range(game.cols):
            if propability_board[row, col] == max_value:
                game.flag_cell(row, col)
                total_prob = np.sum(propability_board)
                if total_prob > 0:
                    print(f"flagged ({row}, {col}) with probability {100 * max_value / total_prob:.4f}%")
                else:
                    print(f"flagged ({row}, {col}) with probability undefined (total probability is zero)")
                return True
    return False