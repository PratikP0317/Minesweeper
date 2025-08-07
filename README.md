# Minesweeper Generator

A simple Minesweeper board generator that works like Google's Minesweeper. Generates boards with safe first click and creates two matrices for ML training.

## Features

- **Google-like Difficulties**: Beginner (9×9, 10 mines), Intermediate (16×16, 40 mines), Expert (16×30, 99 mines)
- **Safe First Click**: First click is always guaranteed to be safe
- **Two Matrix Output**: Game board and mine board for ML training
- **Click Simulation**: Simulate revealing cells with flood fill
- **Playable Games**: Test the generator with console or Pygame versions
- **ML Training Ready**: Use with `render=False` for fast automated training

## Matrix Encoding

### Game Board
- `-1`: Unrevealed cell
- `0-8`: Revealed cell with number of adjacent mines

### Mine Board (Binary)
- `0`: Safe cell
- `1`: Mine

## Quick Start

```python
from minesweeper_generator import MinesweeperGenerator

# Initialize generator
generator = MinesweeperGenerator()

# Generate a single board
game_board, mine_board, number_board = generator.generate_board('beginner', first_click=(4, 4))

# Click a cell
revealed_board = generator.click_cell(game_board, number_board, 4, 4)

# Generate training data
game_boards, mine_boards = generator.generate_training_data(1000, 'beginner')

# Create ML dataset
X, y = generator.create_ml_dataset(1000, 'beginner', first_click=(4, 4))
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Generate Single Board
```python
generator = MinesweeperGenerator()
game_board, mine_board, number_board = generator.generate_board('beginner', first_click=(4, 4))

print(f"Game board shape: {game_board.shape}")
print(f"Mine board shape: {mine_board.shape}")
print(f"Number of mines: {np.sum(mine_board)}")
```

### Generate Training Data
```python
game_boards, mine_boards = generator.generate_training_data(1000, 'beginner')
print(f"Generated {len(game_boards)} boards")
print(f"Game boards shape: {game_boards.shape}")
print(f"Mine boards shape: {mine_boards.shape}")
```

### Create ML Dataset
```python
X, y = generator.create_ml_dataset(1000, 'beginner', first_click=(4, 4))
print(f"ML dataset shapes: X={X.shape}, y={y.shape}")
```

## Gym Environment for RL Training

The `MinesweeperGame` class is a full `gymnasium.Env` environment, making it compatible with standard RL libraries:

```python
from play_minesweeper_pygame import MinesweeperGame

# Create the environment
env = MinesweeperGame(difficulty='beginner', render=False)

# Reset to start new episode
obs, info = env.reset()

# Take actions using the gym interface
action = env.action_space.sample()  # Random action
obs, reward, done, truncated, info = env.step(action)

print(f"Observation shape: {obs.shape}")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Won: {info['won']}")
```

### Gym Environment Interface

**Action Space**: `MultiDiscrete([rows, cols])`
- Actions are (row, col) coordinates to click
- Example: `[4, 3]` clicks cell at row 4, column 3

**Observation Space**: `Box(-1, 8, (rows, cols), int32)`
- Current revealed board (-1 for unrevealed, 0-8 for revealed numbers)
- Shape matches board size (9×9 for beginner, etc.)

**Reward Function**:
- **Win**: +100 points
- **Lose**: -50 points
- **Reveal cells**: +1 point per cell revealed

### Standard Gym Methods

```python
# Reset environment
obs, info = env.reset()

# Take action
obs, reward, done, truncated, info = env.step(action)

# Get action/observation spaces
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
```

### Game State for RL Training

The `get_state()` method returns a dictionary with:
- `revealed_board`: Current game board (-1 for unrevealed, 0-8 for revealed)
- `game_over`: Whether the game has ended
- `won`: Whether the game was won
- `mines_remaining`: Number of mines minus flags placed
- `cells_revealed`: Number of cells currently revealed

### Reward Function

The `calculate_reward()` method provides rewards for RL training:
- **Win**: +100 points
- **Lose**: -50 points  
- **Reveal cells**: +1 point per cell revealed

### RL Training Example

```python
# Run multiple games for training
num_games = 1000
total_reward = 0

for game_num in range(num_games):
    game = MinesweeperGame(difficulty='beginner', render=False)
    
    # Your RL agent makes moves here
    while not game.get_state()['game_over']:
        # Get current state for your agent
        state = game.get_state()
        
        # Your agent chooses action
        # action = agent.choose_action(state['revealed_board'])
        
        # Make the move
        # game.click_cell(*action)
        
        # Get reward and learn
        # reward = game.calculate_reward(game.get_state())
        # agent.learn(state, action, reward)
        
        # For this example, just make random moves
        import random
        unrevealed = np.where(state['revealed_board'] == -1)
        if len(unrevealed[0]) > 0:
            idx = random.randint(0, len(unrevealed[0]) - 1)
            game.click_cell(unrevealed[0][idx], unrevealed[1][idx])
    
    final_state = game.get_state()
    total_reward += game.calculate_reward(final_state)

print(f"Average reward per game: {total_reward/num_games:.1f}")
```

## Difficulty Levels

| Difficulty | Board Size | Mines |
|------------|------------|-------|
| Beginner   | 9×9        | 10    |
| Intermediate | 16×16      | 40    |
| Expert     | 16×30      | 99    |

## Running Examples

```bash
# Run the main generator test
python minesweeper_generator.py

# Run the example script
python example.py

# Play the console game to test the generator
python play_minesweeper.py

# Play the Pygame version (visual)
python play_minesweeper_pygame.py

# Run gym environment example
python gym_example.py
```

## Playing the Games

### Console Version (`play_minesweeper.py`)
Simple console-based Minesweeper game:

```bash
python play_minesweeper.py
```

**Commands:**
- `c <row> <col>` - Click a cell
- `f <row> <col>` - Flag/unflag a cell  
- `h` - Show help
- `q` - Quit game

**Board Symbols:**
- `.` - Unrevealed cell
- `F` - Flagged cell
- `0-8` - Number of adjacent mines
- `*` - Mine (shown when game over)

### Pygame Version (`play_minesweeper_pygame.py`)
Visual Minesweeper game with mouse controls:

```bash
python play_minesweeper_pygame.py
```

**Controls:**
- **Left click** - Reveal cell
- **Right click** - Flag/unflag cell
- **R** - Restart game
- **Q** - Quit game

**Features:**
- Clean visual interface
- Color-coded numbers
- Real-time game status
- Easy mouse interaction
- Works with all difficulties
- **ML Training Mode**: Use `render=False` for automated training

## File Structure

```
minesweeper/
├── minesweeper_generator.py  # Main generator class
├── example.py               # Example usage
├── play_minesweeper.py      # Console game to test generator
├── play_minesweeper_pygame.py # Visual Pygame game (ML-ready)
├── ml_training_example.py   # ML training example
├── requirements.txt         # Python dependencies
└── README.md              # This file
```

Perfect for ML training with clean, simple matrix outputs! 