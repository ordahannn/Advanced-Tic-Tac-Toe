#!/usr/bin/env python3
import os, sys, pygame
from math    import inf as infinity
from random  import choice

# ——— CACHE ──────────────────────────────────────────────────────────────────
from functools import lru_cache
import time

# ——— CONFIG ——————————————————————————————————————————————————————————————————
BOARD_SIZE    = 5
CELL_SIZE     = 150
LINE_WIDTH    = 5
HEADER_H      = 100
PROMPT_H      = 50
TOP_BAR       = HEADER_H + PROMPT_H
GRID_PAD      = 20
SCREEN_W      = BOARD_SIZE * CELL_SIZE + GRID_PAD*2   # 790
SCREEN_H      = BOARD_SIZE * CELL_SIZE + TOP_BAR + GRID_PAD*2  # 940
FPS           = 30

GRID_COLOR    = (199,182,160)   # #C7B6A0
BARRIER_BG    = (227,217,205)   # Barrier
GRAYED_OUT = (180, 180, 180)

HUMAN, COMP   = -1, +1
ACT_PLACE     = 'place'
ACT_ERASE     = 'erase'    # Remove a barrier
ACT_BOMB      = 'bomb'     # Clear row+col

ICON_SZ       = 32

ALL_LINES = []
for i in range(BOARD_SIZE):
    ALL_LINES.append([(i, j) for j in range(BOARD_SIZE)])
    ALL_LINES.append([(j, i) for j in range(BOARD_SIZE)])
ALL_LINES.append([(i, i) for i in range(BOARD_SIZE)])
ALL_LINES.append([(i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)])

last_move_time = 0.5

# ——— Placeholders for icon‐hit Rects (initialized after pygame.init) ——————————————————————————————————————————————————————————
ER_BTN = None
BM_BTN = None

# ——— GAMESTATE ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class GameState:
    def __init__(self, board, hb, cb, hd, cd):
        self.board = board  # list of lists (mutable) 0=empty, 2=barrier, -1/1 = X/O
        self.human_bomb = hb      # Bombs left to clear rows+cols
        self.comp_bomb  = cb
        self.human_del  = hd      # Erasers left to remove barriers
        self.comp_del   = cd

    def copy(self):
        new_board = [list(row) for row in self.board]
        return GameState(new_board, self.human_bomb, self.comp_bomb, self.human_del, self.comp_del)

def serialize_state(state):
    flat_board = tuple(cell for row in state.board for cell in row)
    return (
        flat_board,
        state.human_bomb,
        state.comp_bomb,
        state.human_del,
        state.comp_del,
    )

def deserialize_state(state_key):
    flat_board, hb, cb, hd, cd = state_key
    board = [list(flat_board[i * BOARD_SIZE:(i + 1) * BOARD_SIZE]) for i in range(BOARD_SIZE)]
    return GameState(board, hb, cb, hd, cd)

# Function for creation of the initial state for the game
def create_initial_state():
    return GameState(
        [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)], hb=1, cb=1, hd=1, cd=1
    )

# Function to return the empty cells that are left in the board
def empty_cells(b):
    return [(i, j) for i, row in enumerate(b) for j, val in enumerate(row) if val == 0]

# ——— Final states —————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Function to check if one of the players won
def wins(board,player):
    for i in range(BOARD_SIZE):
        row = [board[i][j] for j in range(BOARD_SIZE)]
        col = [board[j][i] for j in range(BOARD_SIZE)]
        if all(cell == player for cell in row) and 2 not in row:
            return True
        if all(cell == player for cell in col) and 2 not in col:
            return True
        
    diag1 = [board[i][i] for i in range(BOARD_SIZE)]
    diag2 = [board[i][BOARD_SIZE - 1 - i] for i in range(BOARD_SIZE)]
    if all (cell == player for cell in diag1) and 2 not in diag1:
        return True
    if all(cell == player for cell in diag2) and 2 not in diag2:
        return True
    
    return False

# Helper to identify the opponent
def opponent(player):
    return HUMAN if player == COMP else COMP

# Check for immediate win/s
def find_immediate_winning_move(state, player, multi=False):
    for act in actions(state, player):
        new_state = result(state, player, act)
        if wins(new_state.board, player):
            if not multi:
                return act
            else:
                win_count = 0
                for next_act in actions(new_state, player):
                    next_state = result(new_state, player, next_act)
                    if wins(next_state.board, player):
                        win_count += 1
                if win_count > 1:
                    return act
    return None

# Function to check if the state is final (no further moves)
def is_terminal(state):
    return wins(state.board, HUMAN) or wins(state.board, COMP) or all(cell != 0 for row in state.board for cell in row)

# ——— EVALUATE MOVES ———————————————————————————————————————————————————————————————————————————————————————————————————————————
def evaluate(state):
    # Terminal wins/losses
    if wins(state.board, COMP): 
        return +100000
    if wins(state.board, HUMAN): 
        return -100000

    score = 0
    weights = {1: 10, 2: 50, 3: 200, 4: 1000}
    
    center_cells = [(2,2), (2,1), (2,3), (1,2), (3,2)]
    
    # Evaluate all lines using evaluate_line_with_barriers for both players
    for line in ALL_LINES:
        line_cells = [state.board[r][c] for r, c in line]
        score += evaluate_line_with_barriers(line_cells, COMP)
        score -= evaluate_line_with_barriers(line_cells, HUMAN)

    # Reward for pieces in and around the center cells
    for r, c in center_cells:
        if state.board[r][c] == COMP:
            score += 15
            # Bonus for neighbors around center cell
            neighbors = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
            for nr,nc in neighbors:
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if state.board[nr][nc] == COMP:
                        score += 3
        elif state.board[r][c] == HUMAN:
            score -= 15
            # Penalty for opponent near center
            neighbors = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
            for nr, nc in neighbors:
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if state.board[nr][nc] == HUMAN:
                        score -= 3

    # Barrier scoring: higher if near active pieces, lower if isolated
    barrier_value = 2.5
    total_barriers = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if state.board[i][j] == 2:
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                close_to_play = any(
                    0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and
                    state.board[x][y] in (HUMAN, COMP)
                    for x, y in neighbors
                )
                if close_to_play:
                    total_barriers += barrier_value
                else:
                    total_barriers += barrier_value * 0.3

    score += total_barriers

    # Reward for special moves left (bombs and erasers)
    token_value = 15
    score += token_value * (state.comp_del - state.human_del)
    score += token_value * (state.comp_bomb - state.human_bomb)

    # Penalty for using special moves too early
    used_erasers = 1 - state.comp_del
    used_bombs = 1 - state.comp_bomb

    early_use_penalty = 0
    if used_erasers > 0:
        early_use_penalty -= used_erasers * 10
    if used_bombs > 0:
        early_use_penalty -= used_bombs * 20

    score += early_use_penalty

    # Additional positional weighting — reward occupying corners and edges less than center
    corner_positions = [(0,0), (0,BOARD_SIZE-1), (BOARD_SIZE-1,0), (BOARD_SIZE-1,BOARD_SIZE-1)]
    edge_positions = [(0,i) for i in range(1,BOARD_SIZE-1)] + [(BOARD_SIZE-1,i) for i in range(1,BOARD_SIZE-1)] + \
                     [(i,0) for i in range(1,BOARD_SIZE-1)] + [(i,BOARD_SIZE-1) for i in range(1,BOARD_SIZE-1)]

    for r,c in corner_positions:
        if state.board[r][c] == COMP:
            score += 7
        elif state.board[r][c] == HUMAN:
            score -= 7

    for r,c in edge_positions:
        if state.board[r][c] == COMP:
            score += 4
        elif state.board[r][c] == HUMAN:
            score -= 4

    return score


def evaluate_line_with_barriers(line_cells, player):
    barrier_positions = [i for i, val in enumerate(line_cells) if val == 2]
    player_count = line_cells.count(player)
    opponent = HUMAN if player == COMP else COMP
    opponent_count = line_cells.count(opponent)
    
    # If opponent's marks are present, the line is blocked and scores zero
    if opponent_count > 0:
        return 0
    
    # Base score according to the count of player's marks in the line
    weights = {0: 0, 1: 10, 2: 50, 3: 200, 4: 1000, 5: 10000}
    base_score = weights.get(player_count, 0)
    
    # If there are no barriers, return the base score directly
    if not barrier_positions:
        return base_score
    
    # Deduct score depending on barrier positions:
    # - Barrier in the middle of the line causes a bigger penalty
    # - Barrier near edges causes a smaller penalty
    mid = len(line_cells) // 2
    penalty = 0
    for pos in barrier_positions:
        dist_to_mid = abs(pos - mid)
        # Example penalty weight: barrier closer to center reduces score more
        penalty += (mid - dist_to_mid + 1) * 100
    
    # Adjust score by subtracting the penalty
    adjusted_score = base_score - penalty
    
    # Do not return negative scores
    return max(adjusted_score, 0)

def lines_with_n_minus_one(state, player, n=4):
    # Returns the number of lines with n-1 of your symbols
    count = 0
    for line in ALL_LINES:
        line_cells = [state.board[r][c] for r, c in line]
        if line_cells.count(player) == n and line_cells.count(0) == (BOARD_SIZE - n):
            count += 1
    return count

def heuristic(action, state, player, new_state=None):
    typ, (r, c) = action
    if new_state is None:
        new_state = result(state, player, action)

    center = BOARD_SIZE // 2

    # Move that leads to an immediate victory
    if wins(new_state.board, player):
        return 100000
    
    # Move that blocks an immediate victory for the opponent
    opp = opponent(player)
    for opp_act in actions(new_state, opp):
        opp_state = result(new_state, opp, opp_act)
        if wins(opp_state.board, opp):
            return 90000
    
    score = 0

    def near_win_count(s, p, n):
        count = 0
        for line in ALL_LINES:
            line_cells = [s.board[r][c] for r, c in line]
            if line_cells.count(p) == n and line_cells.count(0) == (BOARD_SIZE - n):
                count += 1
        return count
    
    
    score += 2000 * near_win_count(new_state, player, BOARD_SIZE - 1)  # Player's almost full line
    score += 500  * near_win_count(new_state, player, BOARD_SIZE - 2)  # Advanced potential
    score += 100  * near_win_count(new_state, player, BOARD_SIZE - 3)  # Initial potential

    score -= 1500 * near_win_count(new_state, opp, BOARD_SIZE - 1)    # Opponent's close threats
    score -= 400  * near_win_count(new_state, opp, BOARD_SIZE - 2)
    
    # Weight for the center of the board and cells around it
    center_positions = [(center, center), (center-1, center), (center+1, center), (center, center-1), (center, center+1)]
    center_weight = 0
    if (r, c) in center_positions:
        center_weight = 50  # Can be adjusted if needed

    # Adjacency weight to the player himself (aggregation on neighbors of the box)
    adj_weight = 0
    for i, j in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and state.board[i][j] == player:
            adj_weight += 40  # High weight for adjacency

    # Barrier evaluation higher weight for barriers that block lines with many signs
    barrier_score = 0
    for line in ALL_LINES:
        line_cells = [new_state.board[r][c] for r, c in line]
        if 2 in line_cells:  # If there is a barrier in the row
            count_player = line_cells.count(player)
            count_opp = line_cells.count(opp)
            # A barrier that blocks almost a full row to the player
            if count_player >= BOARD_SIZE - 2:
                barrier_score += 100 * count_player
            # A barrier that blocks almost a full row to the opponent
            if count_opp >= BOARD_SIZE - 2:
                barrier_score += 150 * count_opp

    # Bomb type moves
    bomb_score = 0
    if typ == ACT_BOMB:
        # Calculate how many symbols can be cleared in a row and column
        row_count = sum(1 for j in range(BOARD_SIZE) if new_state.board[r][j] in (HUMAN, COMP))
        col_count = sum(1 for i in range(BOARD_SIZE) if new_state.board[i][c] in (HUMAN, COMP))
        bomb_score = 50 * (row_count + col_count)  # High weight for large cleanups

    # Basic score for distance to the center and adjacency
    dist_to_center = abs(r - center) + abs(c - center)
    base_score = 500 - dist_to_center * 10 + adj_weight

    # Deletion type moves
    erase_score = 0
    if typ == ACT_ERASE:
        row_marks = sum(1 for j in range(BOARD_SIZE) if new_state.board[r][j] == player)
        col_marks = sum(1 for i in range(BOARD_SIZE) if new_state.board[i][c] == player)
        erase_score = (row_marks + col_marks) * 30

    # Balance between attack and defense (weights can be changed dynamically according to the game situation)
    final_score = base_score + barrier_score + bomb_score + erase_score + center_weight

    return final_score
    
# ——— RESULTS ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def result(state, player, act):
    typ, (r,c) = act

    # Copy the board
    board_copy = [row[:] for row in state.board]
    new_state = GameState(
        board=board_copy,
        hb=state.human_bomb,
        cb=state.comp_bomb,
        hd=state.human_del,
        cd=state.comp_del,
    )

    if typ == ACT_PLACE: # Place X/O
        board_copy[r][c] = player

    elif typ == ACT_ERASE: # Erase barrier
        if board_copy[r][c] == 2:
            board_copy[r][c] = 0
            if player == COMP:
                new_state.comp_del -= 1
            else:
                new_state.human_del -= 1
    
    else: # Bomb row+column of a cell (clears only X/O)
        for j in range(BOARD_SIZE):
            if board_copy[r][j] in (HUMAN, COMP):
                board_copy[r][j] = 0
        for i in range(BOARD_SIZE):
            if board_copy[i][c] in (HUMAN, COMP):
                board_copy[i][c] = 0
        if player == COMP:
            new_state.comp_bomb -= 1
        else:
            new_state.human_bomb -= 1
    return new_state

# ——— MINIMAX + ALPHA BETA PRUNING —————————————————————————————————————————————————————————————————————————————————————————————
@lru_cache(maxsize = None)
def minimax_cached(s_repr, player, α, β, depth):
    state = deserialize_state(s_repr)

    immediate_win = find_immediate_winning_move(state, player, multi=False)
    if immediate_win is not None:
        return immediate_win, 10000 if player == COMP else -10000

    immediate_multi_win = find_immediate_winning_move(state, player, multi=True)
    if immediate_multi_win is not None:
        return immediate_multi_win, 10000 if player == COMP else -10000

    immediate_block = find_immediate_winning_move(state, opponent(player), multi=False)
    if immediate_block is not None:
        return immediate_block, 9000 if player == COMP else -9000

    if depth == 0 or is_terminal(state):
        return None, evaluate(state)
    
    best_act, best_val = (None, -infinity) if player == COMP else (None, infinity)

    raw_actions = actions(state, player)

    # Prepare actions and their resulting states to avoid recalculations
    actions_and_states = [(act, result(state, player, act)) for act in raw_actions]

    # Check for immediate wins again on these results (optional but safer)
    for act, new_state in actions_and_states:
        if wins(new_state.board, player):
            return act, 10000 if player == COMP else -10000

    # Sort the possible moves by heuristic
    sorted_actions = sorted(
        actions_and_states,
        key=lambda x: heuristic(x[0], state, player, new_state=x[1]),
        reverse=True
    )

    for act, new_state in sorted_actions:
        new_s_repr = serialize_state(new_state)

        _, v = minimax_cached(new_s_repr, opponent(player), α, β, depth - 1)

        if player == COMP and v > best_val:
            best_val, best_act = v, act
        if player == HUMAN and v < best_val:
            best_val, best_act = v, act

        if player == COMP:
            α = max(α, best_val)
        else:
            β = min(β, best_val)

        if β <= α:
            break

    return best_act, best_val

# Function for choosing the depth
def adaptive_depth(state, last_move_time):
    empty = len(empty_cells(state.board))
    specials_left = (state.comp_bomb + state.comp_del + state.human_bomb + state.human_del)

    base_depth = 0
    if empty > 16:
        base_depth = 3
    elif empty > 10:
        base_depth = 4
    elif empty > 6:
        base_depth = 5
    else:
        base_depth = 6

    # Consider the number of special moves left
    if specials_left <= 2:
        base_depth = max(2, base_depth - 1)

    # Reduce depth if the time to calculate the previous move was long
    if last_move_time > 2.0:
        base_depth = max(2, base_depth - 1)

    # Add another cut if the time is really long
    if last_move_time > 5.0:
        base_depth = max(1, base_depth - 2)
    return base_depth

# ——— POSSIBLE MOVES ———————————————————————————————————————————————————————————————————————————————————————————————————————————
def actions(state, player):
    board = state.board
    A = []

    # Empty cell analysis with potential scoring
    candidates = {}

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                score = 0
                # How many symbols are there in a row and column? (potential impact)
                row_marks = sum(1 for x in range(BOARD_SIZE) if board[i][x] in (HUMAN, COMP))
                col_marks = sum(1 for y in range(BOARD_SIZE) if board[y][j] in (HUMAN, COMP))
                score += row_marks + col_marks

                # Extra score if near the center
                center = BOARD_SIZE // 2
                dist_to_center = abs(i - center) + abs(j - center)
                score += max(0, 3 - dist_to_center) 

                candidates[(i,j)] = score
        
    # Choose the moves with a higher score
    if candidates:
        threshold = sorted(candidates.values(), reverse=True)[max(0, len(candidates) // 5)]
        best_candidates = [pos for pos, sc in candidates.items() if sc >= threshold]
    else:
        best_candidates = []
        
    # Adding moves with good scoring
    for pos in best_candidates:
        A.append((ACT_PLACE, pos))

    # If there are no good candidates (empty board) add all empty spots
    if not A:
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == 0:
                    A.append((ACT_PLACE, (i,j)))

    erasers = state.comp_del if player == COMP else state.human_del
    if erasers > 0:
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == 2:
                    A.append((ACT_ERASE, (i, j)))

    bombs = state.comp_bomb if player == COMP else state.human_bomb
    if bombs > 0:
        targets_in_rows = [any(board[r][x] in (HUMAN, COMP) for x in range(BOARD_SIZE)) for r in range(BOARD_SIZE)]
        targets_in_cols = [any(board[y][c] in (HUMAN, COMP) for y in range(BOARD_SIZE)) for c in range(BOARD_SIZE)]

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if targets_in_rows[r] or targets_in_cols[c]:
                    A.append((ACT_BOMB, (r, c)))

    return A

# ——— ASSETS ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def load_assets():
    I = os.path.join("images", "icons")
    S = os.path.join("images", "screens")

    # Icons
    x_img = pygame.image.load(os.path.join(I, "x.png"     )).convert_alpha()
    o_img = pygame.image.load(os.path.join(I, "o.png"     )).convert_alpha()
    e_ic  = pygame.image.load(os.path.join(I, "eraser.png")).convert_alpha()
    b_ic  = pygame.image.load(os.path.join(I, "bomb.png"  )).convert_alpha()
    x_img = pygame.transform.smoothscale(x_img, (CELL_SIZE, CELL_SIZE))
    o_img = pygame.transform.smoothscale(o_img, (CELL_SIZE, CELL_SIZE))
    e_ic  = pygame.transform.smoothscale(e_ic, (ICON_SZ, ICON_SZ))
    b_ic  = pygame.transform.smoothscale(b_ic, (ICON_SZ, ICON_SZ))
    
    # Full‐screen Canva backgrounds (must be 790×940)
    screens = {}
    for key,fn in [
      ("start", "start_screen.png"),
      ("player_barrier", "player_barrier_screen.png"),
      ("ai_barrier", "ai_barrier_screen.png"),
      ("header", "none_screen.png"),
      ("none", "none_screen.png"),
      ("erase", "eraser_screen.png"),
      ("bomb", "bomb_screen.png"),
      ("ai_think", "ai_screen.png"),
      ("win", "win_screen.png"),
      ("lose", "lose_screen.png"),
      ("tie", "tie_screen.png"),
    ]:
        screens[key] = pygame.image.load(os.path.join(S, fn)).convert()
    return x_img, o_img, e_ic, b_ic, screens


def wait_click():
    while True:
        ev = pygame.event.wait()
        if ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            return
        if ev.type == pygame.QUIT:
            pygame.quit(); sys.exit()

# ——— DRAW HELPERS —————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Function to create the default background
def draw_background(scr, screens, mode):
    scr.blit(screens.get(mode, "none"), (0, 0))

def draw_header(scr, screens, e_ic, b_ic, state):
    fnt = pygame.font.SysFont(None, 28)
    y = (HEADER_H - ICON_SZ) // 2

    inactive = (150, 150, 150)
    active   = (50, 50, 50)

    # Human
    lx = GRID_PAD
    er_color = inactive if state.human_del == 0 else active
    scr.blit(e_ic, (lx, y))
    scr.blit(fnt.render(str(state.human_del), True, er_color), (lx + ICON_SZ + 4, y))

    bx = lx + 48
    bm_color = inactive if state.human_bomb == 0 else active
    scr.blit(b_ic, (bx, y))
    scr.blit(fnt.render(str(state.human_bomb), True, bm_color), (bx + ICON_SZ + 4, y))

    # AI
    rbx = SCREEN_W - GRID_PAD - ICON_SZ
    scr.blit(b_ic, (rbx, y))
    scr.blit(fnt.render(str(state.comp_bomb), True, active), (rbx + ICON_SZ + 4, y))

    rex = rbx - 48
    scr.blit(e_ic, (rex, y))
    scr.blit(fnt.render(str(state.comp_del), True, active), (rex + ICON_SZ + 4, y))

# Function to create the grid of the GameBoard
def draw_grid(scr):
    x0, x1 = GRID_PAD, SCREEN_W - GRID_PAD
    y0, y1 = TOP_BAR + GRID_PAD, SCREEN_H - GRID_PAD
    for i in range(BOARD_SIZE+1):
        x = x0 + i * CELL_SIZE
        pygame.draw.line(scr, GRID_COLOR, (x, y0), (x, y1), LINE_WIDTH)
    for j in range(BOARD_SIZE+1):
        y = y0 + j * CELL_SIZE
        pygame.draw.line(scr, GRID_COLOR, (x0, y), (x1, y), LINE_WIDTH)

# Function to place the X/O's
def draw_marks(scr, state, x_img, o_img):
    y0 = TOP_BAR + GRID_PAD
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            v = state.board[i][j]
            px = GRID_PAD + j * CELL_SIZE
            py = y0 + i * CELL_SIZE
            if v == HUMAN:
                scr.blit(x_img, (px, py))
            elif v == COMP:
                scr.blit(o_img, (px, py))
            elif v == 2:
                pygame.draw.rect(scr, BARRIER_BG,
                    (px + LINE_WIDTH, py + LINE_WIDTH,
                     CELL_SIZE - 2 * LINE_WIDTH, CELL_SIZE - 2 * LINE_WIDTH))

# ——— BARRIER PLACEMENT ————————————————————————————————————————————————————————————————————————————————————————————————————————
def place_barriers(screen, screens, state, x_img, o_img, e_ic, b_ic):
    f = pygame.font.SysFont(None, 32)
    for who in (HUMAN, COMP):
        placed = False
        key = 'player_barrier' if who == HUMAN else 'ai_barrier'
        label = ""
        while not placed:
            # Show the full-screen images
            draw_background(screen, screens, key)

            # Overlay just the icons/counts (not entire header)
            draw_header(screen, screens, e_ic, b_ic, state)

            # Draw our grid & any existing marks/barriers
            txt = f.render(label, True, (50,50,50))
            screen.blit(txt, (20, HEADER_H + (PROMPT_H - txt.get_height()) // 2))
            
            draw_grid(screen)
            draw_marks(screen, state, x_img, o_img)
            
            pygame.display.flip()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if who == HUMAN and ev.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = ev.pos
                    x0,y0 = GRID_PAD, TOP_BAR+GRID_PAD
                    if x0 <= mx < SCREEN_W-GRID_PAD and y0 <= my < SCREEN_H-GRID_PAD:
                        r = (my - y0) // CELL_SIZE
                        c = (mx - x0) // CELL_SIZE
                        if state.board[r][c] == 0:
                            state.board[r][c] = 2
                            placed = True

            if who == COMP:
                i,j = choice(empty_cells(state.board))
                state.board[i][j] = 2
                placed = True

            pygame.time.wait(200)

# ——— MAIN LOOP ————————————————————————————————————————————————————————————————————————————————————————————————————————————————
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Tic-Tac-Toe")
    clock  = pygame.time.Clock()

    # Now pygame is inited, build the icon rects
    ER_BTN = pygame.Rect(20, (HEADER_H-ICON_SZ) // 2, ICON_SZ, ICON_SZ)
    BM_BTN = pygame.Rect(20 + 48, (HEADER_H-ICON_SZ) // 2, ICON_SZ, ICON_SZ)

    x_img, o_img, e_ic, b_ic, screens = load_assets()

    # Title
    draw_background(screen, screens, 'start')
    pygame.display.flip()
    wait_click()

    # Barrier placement
    state = create_initial_state()
    place_barriers(screen, screens, state, x_img, o_img, e_ic, b_ic)

    # Gameplay
    player, mode = HUMAN, None
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.MOUSEBUTTONDOWN and player == HUMAN:
                mx,my = ev.pos
                if mode is None:
                    if ER_BTN.collidepoint(mx, my) and state.human_del > 0:
                        mode = 'erase'
                    elif BM_BTN.collidepoint(mx, my) and state.human_bomb > 0:
                        mode = 'bomb'
                else:
                    if mode =='erase' and ER_BTN.collidepoint(mx, my):
                        mode = None
                    if mode =='bomb' and BM_BTN.collidepoint(mx, my):
                        mode = None

            if (ev.type == pygame.MOUSEBUTTONDOWN and player == HUMAN and mode is None):
                mx, my = ev.pos
                x0, y0 = GRID_PAD, TOP_BAR+GRID_PAD
                if x0 <= mx < SCREEN_W-GRID_PAD and y0 <= my < SCREEN_H - GRID_PAD:
                    r = (my - y0) // CELL_SIZE; c = (mx-x0) // CELL_SIZE
                    if state.board[r][c] == 0:
                        state = result(state, HUMAN, (ACT_PLACE, (r, c)))
                        player = COMP

            # Erase or bomb click
            if (ev.type == pygame.MOUSEBUTTONDOWN and player == HUMAN and mode in ('erase','bomb')):
                mx, my = ev.pos
                x0, y0 = GRID_PAD, TOP_BAR + GRID_PAD
                if x0 <= mx < SCREEN_W-GRID_PAD and y0 <= my < SCREEN_H-GRID_PAD:
                    r = (my - y0) // CELL_SIZE; c = (mx - x0) // CELL_SIZE
                    if mode == 'erase' and state.board[r][c] == 2:
                        state, player, mode = result(state, HUMAN, (ACT_ERASE, (r, c))), COMP, None
                    elif mode == 'bomb':
                        state, player, mode = result(state, HUMAN, (ACT_BOMB, (r, c))), COMP, None

        # AI turn
        if player == COMP and not is_terminal(state):
            # Show thinking screen
            draw_background(screen,screens, 'ai_think')
            draw_header(screen, screens, e_ic, b_ic, state)
            draw_grid(screen)
            draw_marks(screen, state, x_img, o_img)
            pygame.display.flip()
            pygame.time.wait(300)

            depth = adaptive_depth(state, last_move_time)

            print(f"\033[90m\n—————————— AI eval (d={depth}) ——————————", "\033[0m")

            start_time = time.perf_counter()

            # act = minimax(state, depth, COMP, -infinity, infinity)[0]
            act, val = minimax_cached(
                s_repr=serialize_state(state),
                player=COMP,
                α=-infinity,
                β=infinity,
                depth=depth
            )

            end_time = time.perf_counter()
            last_move_time = end_time - start_time
            
            print(f"\033[94m——> AI plays {act} with value {val}", "\033[0m")
            print(f"    Move computed in {end_time - start_time:.4f} seconds")
            print()
            print("\033[92mCache stats:", minimax_cached.cache_info(), "\033[0m")
            
            # New default state after AI move
            state = result(state, COMP, act)
            player = HUMAN
            mode = None

        # Draw main
        draw_background(screen, screens, mode or 'header')
        draw_header(screen, screens, e_ic, b_ic, state)
        draw_grid(screen)
        draw_marks(screen, state, x_img, o_img)
        pygame.display.flip()
        clock.tick(FPS)

        # End‐of‐game?
        if is_terminal(state):
            key = 'tie'
            if wins(state.board, HUMAN): 
                key='win'
            if wins(state.board, COMP): 
                key='lose'
            draw_background(screen, screens, key)
            pygame.display.flip()
            wait_click()
            
            # Restart
            state = create_initial_state() # Defaukt state
            player = HUMAN # First player is always the human (can be adjusted)
            mode = None # Special action
            place_barriers(screen, screens, state, x_img, o_img, e_ic, b_ic)