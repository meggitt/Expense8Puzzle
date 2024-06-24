# Expense8Puzzle: A Python Implementation for Solving 8-Puzzle Problem

## Overview
This repository contains a Python implementation of various algorithms to solve the classic 8-puzzle problem. The algorithms implemented include:

- Breadth-First Search (BFS)
- Uniform Cost Search (UCS)
- Depth-First Search (DFS)
- Depth-Limited Search (DLS)
- Iterative Deepening Search (IDS)
- Greedy Best-First Search
- A* Search

The program reads the initial and goal states of the puzzle from input files and finds the sequence of moves to solve the puzzle using the selected algorithm.

## Requirements

- Python v3.0 or greater (Download: [Python Downloads](https://www.python.org/downloads/))
  - Verify installation by running the following command in Command Prompt:
    ```bash
    python --version
    ```
    The output should be the installed Python version, for example:
    ```
    Python 3.10.11
    ```
- (Optional) Any IDE

## Files

- `expense_8_puzzle.py`: Main implementation file containing the algorithms and utility functions.
- `start.txt`: Example file containing the initial state of the puzzle.
- `goal.txt`: Example file containing the goal state of the puzzle.

## Classes and Functions

### Class `PuzzleTile`

This class has the following methods:

- `generateHeuristic`: Generates the heuristic by calculating the distance from the state of the tile to its goal state.
- `generateHeuristicGreedy`: Generates heuristic for greedy search by adding the above heuristic to the cost.
- `isGoalState`: Checks if the current state is the goal state.
- `moves`: Returns a list of valid move directions ('Left', 'Up', 'Right', 'Down') based on the given tile's position in the puzzle.
- `generateSuccessorNodes`: Generates child states and returns them.
- `gofn`: Calculates the total cost of the state and its ancestors.
- `finalMove`: Returns a list of solution nodes, cost, and moves.

### Functions

- `printSolution`: Prints solution information and returns a list of moves.
- `BFS`: Performs Breadth-First Search on the puzzle starting from the initial state. Uses a queue to explore states and returns a solution path.
- `UCS`: Performs Uniform Cost Search on the puzzle starting from the initial state. Uses a priority queue with cost as the priority and returns a solution path.
- `DFS`: Performs Depth-First Search on the puzzle starting from the initial state. Uses a stack (LifoQueue) to explore states and returns a solution path.
- `DLS`: Performs Depth-Limited Search on the puzzle starting from the initial state with a specified depth limit. Uses a stack (LifoQueue) to explore states and returns a solution path.
- `IDS`: Performs Iterative Deepening Search on the puzzle starting from the initial state. Repeatedly performs Depth-Limited Search with increasing depth limits until a solution is found.
- `Greedy`: Performs Greedy Best-First Search on the puzzle starting from the initial state. Uses a priority queue with a heuristic as the priority and returns a solution path.
- `AStar`: Performs A* Search on the puzzle starting from the initial state. Uses a priority queue with the sum of cost and heuristic as the priority and returns a solution path.

### `__main__`

Reads contents from `start.txt` and `goal.txt` files, executes the method mentioned in command line arguments, writes data to the trace log based on the dump flag, and prints the output.

## Execution

To run the program, use the following command:

### Through Command Prompt

- Open the directory where the files are located.
- Execute the following command to compile the code:
  ```bash
  py puzzle_solver.py start.txt goal.txt [method] [isDumpFileRequired]
  ```
  Example:
  ```bash
  py puzzle_solver.py start.txt goal.txt bfs true
  ```

- `start.txt`: File containing the initial state of the puzzle.
- `goal.txt`: File containing the goal state of the puzzle.
- `[method]`: (Optional) Algorithm to use. Default is "a*". Options include "bfs", "ucs", "dfs", "dls", "ids", "greedy", "a*".
- `[isDumpFileRequired]`: (Optional) If "true", generates a trace file with the detailed execution log. Default is "false".

### Example

```bash
py puzzle_solver.py start.txt goal.txt bfs true
```

This command will use the BFS algorithm to solve the puzzle and generate a trace file.

## Output

The program outputs the steps to solve the puzzle, including the number of nodes popped, nodes expanded, nodes generated, and the maximum fringe size.

### Example Output

```
Nodes Popped = 1136
Nodes Expanded = 1045
Nodes Generated = 2777
Max Fringe Size= 593
Solution found at depth 12 with cost of 63
Steps:
    Move 7 Left
    Move 5 Up
    Move 8 Right
    Move 7 Down
    Move 5 Left
    Move 6 Down
    Move 3 Right
    Move 2 Right
    Move 1 Up
    Move 4 Up
    Move 7 Left
    Move 8 Left
    ...
```

### Using Visual Studio Code

- Install and configure Python ([VSCode Python Configuration](https://code.visualstudio.com/docs/languages/python)).
- Click on `Terminal -> New Terminal` and execute the command:
  ```bash
  py expense_8_puzzle.py start.txt goal.txt <method> <isDumpFileRequired>
  ```
  Example:
  ```bash
  py expense_8_puzzle.py start.txt goal.txt bfs true
  ```

## Behavior

- If no method is mentioned, `A*` is executed.
- If no boolean dump value is provided, it is treated as `false` for the dump.
- If no method is mentioned and the dump flag is provided, `A*` with the specified dump flag is executed.
- If no method is mentioned and no dump flag is provided, `A*` with the dump flag set to `false` is executed.
- DFS may take a long time to find the solution.
- For DLS, provide a depth of 50 or more to get the solution.