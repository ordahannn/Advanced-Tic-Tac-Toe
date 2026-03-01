# Web Strategy Game – Final Project for Advanced Algorithms

---

## Project Overview

This project presents an **enhanced and intelligent version of the classic Tic-Tac-Toe game**, developed as part of our final assignment for the Advanced Algorithms for Intelligent Systems course.

Unlike the original game, this version is played on a **5×5 board** and introduces **strategic mechanics** such as barrier placement, bombs, and row/column erasure — transforming the game into a deeper, more complex challenge where every move matters.

The computer opponent is powered by a **Minimax algorithm with Alpha-Beta pruning**, enhanced with LRU caching, adaptive search depth, and a custom heuristic evaluation function.

---

## Main Screens
- **Game States**

  <img width="320" height="385" alt="image" src="https://github.com/user-attachments/assets/4d331045-a237-48ca-8e03-9e0a05eeec3f" />
  <img width="320" height="385" alt="image" src="https://github.com/user-attachments/assets/fb6654b2-eb20-4992-8e17-7b7a9c1c2cc7" />

- **Game Actions**

  <img width="320" height="385" alt="image" src="https://github.com/user-attachments/assets/83beb94e-9fa0-49ca-b854-069c89c226ca" />
  <img width="320" height="385" alt="image" src="https://github.com/user-attachments/assets/b9ab041d-4c7a-41a0-966a-5ae8fe96d814" />
  <img width="320" height="385" alt="image" src="https://github.com/user-attachments/assets/b213d01c-9e56-4290-9122-53831f9b7b90" />
    
- **Game Results**

  <img width="320" height="385" alt="image" src="https://github.com/user-attachments/assets/c32f770d-b0ee-4f4b-9843-1f7c6e0bb86b" />
  <img width="320" height="385" alt="image" src="https://github.com/user-attachments/assets/befd8d16-bbb7-4705-adf1-dba01f21ecb4" />
  <img width="320" height="385" alt="image" src="https://github.com/user-attachments/assets/98656fba-b513-4a15-a752-8bfa28ef1b8c" />

---

## Game Principles

- **Two players**: Human (X) vs. AI (O), taking turns.
- **Board size**: 5×5.
- **Winning condition**: Be the first to fill an **entire row, column, or diagonal** with 5 of your own symbols (barriers do not count toward a win).

---

## Unique Features

### Pre-game Phase – Barrier Placement
- Each player places **1 barrier** on the board before the first move, alternating turns (Human first, then AI).
- Barriers **block access** to the cell they occupy — neither player can place a symbol on a barrier cell.
- Barriers persist throughout the game unless removed by an eraser.

### In-game Special Actions

Each player starts the game with **1 bomb** and **1 eraser**.

**Bomb**
- Targets a row and column simultaneously.
- Clears **all player symbols (X and O)** in the selected row and column.
- Barriers are **not** affected by bombs.
- Each player has **1 bomb** per game.

**Eraser**
- Removes **1 barrier** from any cell on the board.
- Each player has **1 eraser** per game.

---

## AI – Minimax with Alpha-Beta Pruning

The AI opponent is built using a full **Minimax algorithm** with the following optimizations:

- **Alpha-Beta Pruning** – Eliminates branches that cannot affect the final decision, significantly reducing computation.
- **LRU Caching** (`@lru_cache`) – Caches previously evaluated game states to avoid redundant calculations.
- **Adaptive Search Depth** – Dynamically adjusts depth based on remaining empty cells and previous move computation time (ranges from depth 3 to depth 6).
- **Move Ordering via Heuristic** – Sorts candidate moves by a scoring function before searching, improving pruning efficiency.
- **Immediate Win/Block Detection** – Checks for winning or blocking moves before entering the full search tree.

### Evaluation Function

The heuristic evaluates board states using:
- Line scoring (rows, columns, diagonals) weighted by symbol count
- Barrier impact on line potential
- Center and corner positional bonuses
- Remaining special moves (bombs/erasers) as resource value
- Penalty for using special moves too early

---

## Strategic Considerations

- Barriers, bombs, and erasers significantly increase **planning depth** — each action has both immediate and long-term consequences.
- The AI evaluates not only piece placement but also when and where to use its special actions.
- The rule system was carefully balanced to ensure the game remains **fair, playable, and strategically rich** for both players.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/ordahannn/Smart-Tic-Tac-Toe.git
cd Smart-Tic-Tac-Toe
pip install pygame
python tic_tac_toe_pygame.py
```

---

## Tech Stack

- **Python**
- **Pygame** – Game rendering and UI
- **functools.lru_cache** – State memoization
- **Custom AI engine** – Minimax + Alpha-Beta + adaptive depth

---

## Repository Structure

```
├── tic_tac_toe_pygame.py   # Main game file (AI engine + game loop + rendering)
└── images/
    ├── icons/              # X, O, eraser, bomb icons
    └── screens/            # Full-screen background images (start, win, lose, etc.)
```

---

## Acknowledgments

This project was developed as part of the **Advanced Algorithms for Intelligent Systems** course at Ruppin Academic Center. Special thanks to our instructor and mentors for their guidance throughout the project.
