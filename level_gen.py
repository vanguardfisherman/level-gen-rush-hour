# genlevels.py
# Generador de niveles "Rush Hour" (6x6) + solver BFS + export a TypeScript LevelDef[]
# Ejecuta:  python genlevels.py --difficulty easy --n-levels 10 > easy.out.ts

import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Sequence

# ---------------------------
# Config base del tablero
# ---------------------------

BOARD_SIZE = 6

# Player fijo, salida fija (coincide con tu juego)
PLAYER_ID = "P"
PLAYER = {"id": "P", "len": 2, "dir": "v", "x": 3, "y": 3, "asset": "player_len2_red"}
EXIT = {"side": "top", "index": 3}

ASSETS_BY_LEN = {
    2: ["car_len2_blue", "car_len2_gray"],
    3: ["car_len3_red", "car_len3_purple", "car_len3_mili"],
    4: ["car_len4_red", "car_len4_yellow", "car_len4_gray"],
}

# Perfiles de dificultad (editables)
PROFILES = {
    "easy": {
        "min_moves": 10, "max_moves": 16,
        "num_pieces_range": (8, 12),
        "cover_top_cells": 2,            # cuántas de (3,2),(3,1),(3,0) intentar cubrir
        "lens_weights": (2, 6, 2),       # pesos para longitudes (2,3,4)
        "dir_bias": (1.0, 1.0),          # (bias_h, bias_v) → 1:1 neutral
        "seed": 2025,
    },
    "normal": {
        "min_moves": 16, "max_moves": 24,
        "num_pieces_range": (9, 13),
        "cover_top_cells": 3,
        "lens_weights": (1, 7, 2),
        "dir_bias": (0.9, 1.1),          # leve sesgo a verticales
        "seed": 2025,
    },
    "hard": {
        "min_moves": 24, "max_moves": 36,
        "num_pieces_range": (10, 14),
        "cover_top_cells": 3,
        "lens_weights": (1, 6, 3),       # más len=4
        "dir_bias": (0.8, 1.2),          # más verticales para “atascar” columna
        "seed": 2025,
    },
}

# ---------------------------
# Modelo de pieza
# ---------------------------

@dataclass
class Piece:
    id: str
    len: int      # 2,3,4
    dir: str      # 'h' or 'v'
    x: int
    y: int
    asset: str

    def copy(self):
        return Piece(self.id, self.len, self.dir, self.x, self.y, self.asset)

def pieces_copy(lst: List[Piece]) -> List[Piece]:
    return [p.copy() for p in lst]

# ---------------------------
# Utilidades de tablero
# ---------------------------

def in_bounds(p: Piece) -> bool:
    if p.dir == 'h':
        return 0 <= p.x <= BOARD_SIZE - p.len and 0 <= p.y < BOARD_SIZE
    else:
        return 0 <= p.y <= BOARD_SIZE - p.len and 0 <= p.x < BOARD_SIZE

def build_grid(pieces: List[Piece], ignore_id: Optional[str] = None) -> List[List[int]]:
    g = [[-1]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    for p in pieces:
        if ignore_id and p.id == ignore_id:
            continue
        if p.dir == 'h':
            for dx in range(p.len):
                g[p.y][p.x+dx] = p.len
        else:
            for dy in range(p.len):
                g[p.y+dy][p.x] = p.len
    return g

def overlap_free(pieces: List[Piece]) -> bool:
    g = [[-1]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    for p in pieces:
        if not in_bounds(p):
            return False
        if p.dir == 'h':
            for dx in range(p.len):
                y, x = p.y, p.x+dx
                if g[y][x] != -1:
                    return False
                g[y][x] = 1
        else:
            for dy in range(p.len):
                y, x = p.y+dy, p.x
                if g[y][x] != -1:
                    return False
                g[y][x] = 1
    return True

# ---------------------------
# Solver BFS (igual a tu TS)
# ---------------------------

def slide_targets(grid: List[List[int]], p: Piece) -> List[int]:
    res = []
    if p.dir == 'h':
        # izquierda
        nx = p.x - 1
        while nx >= 0 and grid[p.y][nx] == -1:
            res.append(nx); nx -= 1
        # derecha (desde la cola)
        nx = p.x + 1
        while (nx + p.len - 1) < BOARD_SIZE and grid[p.y][nx + p.len - 1] == -1:
            res.append(nx); nx += 1
    else:
        # arriba
        ny = p.y - 1
        while ny >= 0 and grid[ny][p.x] == -1:
            res.append(ny); ny -= 1
        # abajo (desde la cola)
        ny = p.y + 1
        while (ny + p.len - 1) < BOARD_SIZE and grid[ny + p.len - 1][p.x] == -1:
            res.append(ny); ny += 1
    return res

def is_goal(pieces: List[Piece]) -> bool:
    # Player vertical, salida top@x=3
    P = next((x for x in pieces if x.id == PLAYER_ID), None)
    if not P: return False
    if P.dir != 'v' or P.x != EXIT["index"]:
        return False
    grid = build_grid(pieces, ignore_id=PLAYER_ID)
    # ¿hay camino libre hacia arriba?
    y = P.y - 1
    while y >= 0:
        if grid[y][P.x] != -1: return False
        y -= 1
    return True

def encode(pieces: List[Piece], order: List[str], pos_idx: Dict[str,int]) -> str:
    key = [0]*(2*len(order))
    for p in pieces:
        i = pos_idx[p.id]*2
        key[i] = p.x
        key[i+1] = p.y
    return ",".join(map(str,key))

def solve_bfs(pieces_start: List[Piece]) -> Optional[Tuple[int, List[Tuple[str, Tuple[int,int], Tuple[int,int]]]]]:
    """ Devuelve (visited, moves) o None si no hay solución.
        moves = [(pieceId, (fromx,fromy), (tox,toy)), ...] (óptimos)
    """
    order = sorted([p.id for p in pieces_start])
    pos_idx = {pid:i for i,pid in enumerate(order)}

    start_key = encode(pieces_start, order, pos_idx)
    q = [start_key]
    by_key: Dict[str,List[Piece]] = { start_key: pieces_copy(pieces_start) }
    parent: Dict[str, Tuple[Optional[str], Optional[Tuple[str, Tuple[int,int], Tuple[int,int]]]]] = {
        start_key: (None, None)
    }

    head = 0
    visited = 0

    while head < len(q):
        key = q[head]; head += 1; visited += 1
        pieces = by_key[key]

        if is_goal(pieces):
            # reconstruir
            moves = []
            cur = key
            while True:
                prev, move = parent[cur]
                if move is not None:
                    moves.append(move)
                if prev is None: break
                cur = prev
            moves.reverse()
            return visited, moves

        # expandir
        for p in pieces:
            grid = build_grid(pieces, ignore_id=p.id)
            targets = slide_targets(grid, p)
            for t in targets:
                nxt = pieces_copy(pieces)
                me = next(x for x in nxt if x.id == p.id)
                frm = (me.x, me.y)
                if me.dir == 'h': me.x = t
                else: me.y = t
                nxt_key = encode(nxt, order, pos_idx)
                if nxt_key in by_key: continue
                by_key[nxt_key] = nxt
                parent[nxt_key] = (key, (p.id, frm, (me.x, me.y)))
                q.append(nxt_key)

    return None

# ---------------------------
# Generación de niveles
# ---------------------------

H = 'h'; V = 'v'

def choose_dir(dir_bias: Tuple[float,float]) -> str:
    bh, bv = dir_bias
    r = random.random() * (bh + bv)
    return H if r < bh else V

def random_piece(pid: str,
                 lens_weights: Tuple[int,int,int] = (2,6,2),
                 dir_bias: Tuple[float,float] = (1.0,1.0)) -> Piece:
    lens = random.choices([2,3,4], weights=lens_weights, k=1)[0]
    dir_ = choose_dir(dir_bias)
    asset = random.choice(ASSETS_BY_LEN[lens])

    if dir_ == H:
        x = random.randint(0, BOARD_SIZE - lens)
        y = random.randint(0, BOARD_SIZE - 1)
    else:
        x = random.randint(0, BOARD_SIZE - 1)
        y = random.randint(0, BOARD_SIZE - lens)

    return Piece(pid, lens, dir_, x, y, asset)

def crosses_cell(p: Piece, cx: int, cy: int) -> bool:
    if p.dir == H:
        return (p.y == cy) and (p.x <= cx <= p.x + p.len - 1)
    else:
        return (p.x == cx) and (p.y <= cy <= p.y + p.len - 1)

def generate_candidate(num_pieces_range=(8,12),
                       cover_top_cells: int = 2,
                       lens_weights: Tuple[int,int,int] = (2,6,2),
                       dir_bias: Tuple[float,float] = (1.0,1.0)) -> List[Piece]:
    """
    Genera una disposición SIN solapes.
    - P fijo en (3,3).
    - 'A' cruza (3,2).
    - Intenta cubrir otras celdas (3,1) y (3,0) según cover_top_cells.
    """
    target = random.randint(*num_pieces_range)
    pieces: List[Piece] = [Piece(**PLAYER)]

    # 1) 'A' cruza (3,2)
    ok = False
    for _ in range(300):
        p = random_piece('A', lens_weights, dir_bias)
        if crosses_cell(p, 3, 2):
            tmp = pieces_copy(pieces) + [p]
            if overlap_free(tmp):
                pieces.append(p); ok = True; break
    if not ok:
        return []

    # helper IDs (evita chr desbordado)
    next_num_id = 1
    def new_id() -> str:
        nonlocal next_num_id
        pid = f"N{next_num_id}"
        next_num_id += 1
        return pid

    # 2) añadir más piezas
    while len(pieces) < (target + 1):
        placed = False
        for _ in range(300):
            pid = new_id()
            q = random_piece(pid, lens_weights, dir_bias)
            tmp = pieces_copy(pieces) + [q]
            if overlap_free(tmp):
                pieces.append(q); placed = True; break
        if not placed:
            return []  # descartar y reintentar nuevo candidato

    # 3) refuerzo de bloqueos en (3,1) y (3,0)
    must_cells = [(3,1), (3,0)]
    random.shuffle(must_cells)
    cells_needed = max(0, min(cover_top_cells-1, 2))  # ya cubrimos (3,2) con 'A'
    for (cx,cy) in must_cells:
        if cells_needed <= 0: break
        if all(not crosses_cell(p, cx, cy) for p in pieces if p.id != PLAYER_ID):
            for _ in range(200):
                pid = new_id()
                q = random_piece(pid, lens_weights, dir_bias)
                if crosses_cell(q, cx, cy):
                    tmp = pieces_copy(pieces) + [q]
                    if overlap_free(tmp):
                        pieces.append(q); cells_needed -= 1; break

    return pieces

def as_ts_level(id_str: str, pieces: List[Piece], difficulty: str) -> str:
    def piece_to_ts(p: Piece) -> str:
        return ("{ id: '%s', len: %d, dir: '%s', x: %d, y: %d, asset: '%s' }"
                % (p.id, p.len, p.dir, p.x, p.y, p.asset))
    pieces_ts = ",\n            ".join(piece_to_ts(p) for p in pieces)
    return f"""\
{{
    id: '{id_str}',
    size: 6,
    difficulty: '{difficulty}',
    exit: {{ side: 'top', index: 3 }},
    pieces: [
            {pieces_ts}
    ],
}}"""

def make_levels(n_levels=10, seed=1234,
                min_moves=10, max_moves=18,
                num_pieces_range=(8,12),
                cover_top_cells=2,
                lens_weights=(2,6,2),
                dir_bias=(1.0,1.0),
                difficulty_label="easy",
                max_attempts=10000) -> List[str]:
    random.seed(seed)
    out_ts_blocks: List[str] = []
    attempts = 0

    while len(out_ts_blocks) < n_levels and attempts < max_attempts:
        attempts += 1
        pieces = generate_candidate(num_pieces_range, cover_top_cells, lens_weights, dir_bias)
        if not pieces:
            continue
        # solver
        res = solve_bfs(pieces)
        if res is None:
            continue
        visited, moves = res
        mcount = len(moves)
        # filtra por dificultad
        if not (min_moves <= mcount <= max_moves):
            continue
        # ordena por id para consistencia visual
        pieces_sorted = sorted(pieces, key=lambda p: (p.id != PLAYER_ID, p.id))
        out_ts_blocks.append(as_ts_level(f"e{len(out_ts_blocks)+1:02d}", pieces_sorted, difficulty_label))

    return out_ts_blocks

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Generador de niveles Rush Hour -> LevelDef[]")
    ap.add_argument("--difficulty", choices=["easy","normal","hard"], default="easy")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    # overrides opcionales
    ap.add_argument("--min-moves", type=int)
    ap.add_argument("--max-moves", type=int)
    ap.add_argument("--min-pieces", type=int)
    ap.add_argument("--max-pieces", type=int)
    ap.add_argument("--cover-top-cells", type=int, choices=[0,1,2,3])
    ap.add_argument("--lens-weights", type=str, help="Pesos '2,6,2' para len (2,3,4)")
    ap.add_argument("--dir-bias", type=str, help="Bias 'h,v' (e.g. '1.0,1.2')")
    ap.add_argument("--outfile", type=str, default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prof = PROFILES[args.difficulty].copy()

    # aplicar overrides CLI
    if args.seed is not None: prof["seed"] = args.seed
    if args.min_moves is not None: prof["min_moves"] = args.min_moves
    if args.max_moves is not None: prof["max_moves"] = args.max_moves
    if args.min_pieces is not None or args.max_pieces is not None:
        lo = args.min_pieces if args.min_pieces is not None else prof["num_pieces_range"][0]
        hi = args.max_pieces if args.max_pieces is not None else prof["num_pieces_range"][1]
        prof["num_pieces_range"] = (lo, hi)
    if args.cover_top_cells is not None:
        prof["cover_top_cells"] = args.cover_top_cells
    if args.lens_weights:
        parts = [int(x) for x in args.lens_weights.split(",")]
        if len(parts) != 3: raise SystemExit("--lens-weights debe ser 'a,b,c' para (len2,len3,len4)")
        prof["lens_weights"] = tuple(parts)
    if args.dir_bias:
        parts = [float(x) for x in args.dir_bias.split(",")]
        if len(parts) != 2: raise SystemExit("--dir-bias debe ser 'bh,bv'")
        prof["dir_bias"] = tuple(parts)

    blocks = make_levels(
        n_levels=args.n_levels,
        seed=prof["seed"],
        min_moves=prof["min_moves"],
        max_moves=prof["max_moves"],
        num_pieces_range=prof["num_pieces_range"],
        cover_top_cells=prof["cover_top_cells"],
        lens_weights=prof["lens_weights"],
        dir_bias=prof["dir_bias"],
        difficulty_label=args.difficulty,
    )

    out = []
    if not blocks:
        out.append("// No se pudieron generar niveles con los parámetros dados.")
    else:
        out.append("// --- PEGAR EN src/game/levels/<diff>.ts ---")
        out.append("import type { LevelDef } from '../types';\n")
        out.append(f"export const {args.difficulty.upper()}_LEVELS: LevelDef[] = [")
        out.append(",\n\n".join(blocks))
        out.append("];")

    text = "\n".join(out)
    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
