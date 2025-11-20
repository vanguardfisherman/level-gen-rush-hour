# genlevels.py
# Generador de niveles "Rush Hour" (tableros configurables) + solver BFS + export a TypeScript LevelDef[]
# Ejecuta:  python genlevels.py --difficulty easy --n-levels 10 > easy.out.ts

import argparse
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------
# Config base del tablero
# ---------------------------

PLAYER_ID = "P"

ASSETS_BY_LEN = {
    2: ["car_len2_blue", "car_len2_gray"],
    3: ["car_len3_red", "car_len3_purple", "car_len3_mili"],
    4: ["car_len4_red", "car_len4_yellow", "car_len4_gray"],
}

# ---------------------------
# Modelo de pieza

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


@dataclass(frozen=True)
class BoardContext:
    board_size: int
    exit_side: str
    exit_index: int
    player: Piece

    def path_cells_to_exit(self) -> List[Tuple[int, int]]:
        if self.exit_side != 'top':
            raise NotImplementedError("Solo se soporta salida superior en esta versión")
        cells: List[Tuple[int, int]] = []
        x = self.exit_index
        y = self.player.y - 1
        while y >= 0:
            cells.append((x, y))
            y -= 1
        return cells


# Perfiles de dificultad (editables)
PROFILES = {
    "easy": {
        "board_size": 6,
        "player": {"id": PLAYER_ID, "len": 2, "dir": "v", "x": 3, "y": 3, "asset": "player_len2_red"},
        "exit": {"side": "top", "index": 3},
        "min_moves": 10,
        "max_moves": 16,
        "num_pieces_range": (8, 12),
        "cover_top_cells": 2,
        "lens_weights": (2, 6, 2),
        "dir_bias": (1.0, 1.0),
        "seed": 2025,
        "level_prefix": "e",
        "max_bfs_states": 200000,
        "max_attempts": 8000,
    },
    "normal": {
        "board_size": 7,
        "player": {"id": PLAYER_ID, "len": 2, "dir": "v", "x": 3, "y": 5, "asset": "player_len2_red"},
        "exit": {"side": "top", "index": 3},
        "min_moves": 20,
        "max_moves": 32,
        "num_pieces_range": (10, 15),
        "cover_top_cells": 4,
        "lens_weights": (1, 7, 3),
        "dir_bias": (0.9, 1.1),
        "seed": 2025,
        "level_prefix": "n",
        "max_bfs_states": 450000,
        "max_attempts": 18000,
    },
    "hard": {
        "board_size": 6,
        "player": {"id": PLAYER_ID, "len": 2, "dir": "v", "x": 3, "y": 3, "asset": "player_len2_red"},
        "exit": {"side": "top", "index": 3},
        "min_moves": 24,
        "max_moves": 36,
        "num_pieces_range": (10, 14),
        "cover_top_cells": 3,
        "lens_weights": (1, 6, 3),
        "dir_bias": (0.8, 1.2),
        "seed": 2025,
        "level_prefix": "h",
        "max_bfs_states": 500000,
        "max_attempts": 18000,
    },
}

# ---------------------------
# Utilidades de tablero
# ---------------------------

def in_bounds(p: Piece, board_size: int) -> bool:
    if p.dir == 'h':
        return 0 <= p.x <= board_size - p.len and 0 <= p.y < board_size
    else:
        return 0 <= p.y <= board_size - p.len and 0 <= p.x < board_size


def build_grid(pieces: List[Piece], board_size: int, ignore_id: Optional[str] = None) -> List[List[int]]:
    g = [[-1] * board_size for _ in range(board_size)]
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


def overlap_free(pieces: List[Piece], board_size: int) -> bool:
    g = [[-1] * board_size for _ in range(board_size)]
    for p in pieces:
        if not in_bounds(p, board_size):
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
    board_size = len(grid)
    if p.dir == 'h':
        # izquierda
        nx = p.x - 1
        while nx >= 0 and grid[p.y][nx] == -1:
            res.append(nx); nx -= 1
        # derecha (desde la cola)
        nx = p.x + 1
        while (nx + p.len - 1) < board_size and grid[p.y][nx + p.len - 1] == -1:
            res.append(nx); nx += 1
    else:
        # arriba
        ny = p.y - 1
        while ny >= 0 and grid[ny][p.x] == -1:
            res.append(ny); ny -= 1
        # abajo (desde la cola)
        ny = p.y + 1
        while (ny + p.len - 1) < board_size and grid[ny + p.len - 1][p.x] == -1:
            res.append(ny); ny += 1
    return res

def is_goal(pieces: List[Piece], ctx: BoardContext) -> bool:
    # Player vertical, salida top
    P = next((x for x in pieces if x.id == PLAYER_ID), None)
    if not P: return False
    if P.dir != 'v' or P.x != ctx.exit_index:
        return False
    grid = build_grid(pieces, ctx.board_size, ignore_id=PLAYER_ID)
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

def solve_bfs(pieces_start: List[Piece], ctx: BoardContext, max_states: int = 300000) -> Optional[Tuple[int, List[Tuple[str, Tuple[int, int], Tuple[int, int]]]]]:
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
        if visited > max_states:
            return None
        pieces = by_key[key]

        if is_goal(pieces, ctx):
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
            grid = build_grid(pieces, ctx.board_size, ignore_id=p.id)
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

def choose_dir(dir_bias: Tuple[float,float], rng: Optional[random.Random] = None) -> str:
    rnd = rng or random
    bh, bv = dir_bias
    r = rnd.random() * (bh + bv)
    return H if r < bh else V

def random_piece(pid: str,
                 board_size: int,
                 lens_weights: Tuple[int, int, int] = (2, 6, 2),
                 dir_bias: Tuple[float, float] = (1.0, 1.0),
                 rng: Optional[random.Random] = None) -> Piece:
    rnd = rng or random
    lens = rnd.choices([2,3,4], weights=lens_weights, k=1)[0]
    dir_ = choose_dir(dir_bias, rnd)
    asset = rnd.choice(ASSETS_BY_LEN[lens])

    if dir_ == H:
        x = rnd.randint(0, board_size - lens)
        y = rnd.randint(0, board_size - 1)
    else:
        x = rnd.randint(0, board_size - 1)
        y = rnd.randint(0, board_size - lens)

    return Piece(pid, lens, dir_, x, y, asset)

def crosses_cell(p: Piece, cx: int, cy: int) -> bool:
    if p.dir == H:
        return (p.y == cy) and (p.x <= cx <= p.x + p.len - 1)
    else:
        return (p.x == cx) and (p.y <= cy <= p.y + p.len - 1)

def generate_candidate(ctx: BoardContext,
                       num_pieces_range: Tuple[int, int] = (8, 12),
                       cover_top_cells: int = 2,
                       lens_weights: Tuple[int, int, int] = (2, 6, 2),
                       dir_bias: Tuple[float, float] = (1.0, 1.0),
                       rng: Optional[random.Random] = None) -> List[Piece]:
    """
    Genera una disposición SIN solapes.
    - P fijo según contexto.
    - 'A' cruza la celda inmediatamente encima del jugador.
    - Intenta cubrir más celdas camino a la salida según cover_top_cells.
    """
    rnd = rng or random

    target = rnd.randint(*num_pieces_range)
    pieces: List[Piece] = [ctx.player.copy()]

    column_cells = ctx.path_cells_to_exit()
    if not column_cells:
        return []

    # 1) 'A' cruza primera celda
    primary_cell = column_cells[0]
    ok = False
    for _ in range(300):
        p = random_piece('A', ctx.board_size, lens_weights, dir_bias, rnd)
        if crosses_cell(p, *primary_cell):
            tmp = pieces_copy(pieces) + [p]
            if overlap_free(tmp, ctx.board_size):
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
            q = random_piece(pid, ctx.board_size, lens_weights, dir_bias, rnd)
            tmp = pieces_copy(pieces) + [q]
            if overlap_free(tmp, ctx.board_size):
                pieces.append(q); placed = True; break
        if not placed:
            return []  # descartar y reintentar nuevo candidato

    # 3) refuerzo de bloqueos sobre la columna de salida
    extra_cells = column_cells[1:]
    rnd.shuffle(extra_cells)
    cells_needed = max(0, min(cover_top_cells - 1, len(extra_cells)))  # ya cubrimos la primera con 'A'
    for (cx, cy) in extra_cells:
        if cells_needed <= 0: break
        if all(not crosses_cell(p, cx, cy) for p in pieces if p.id != PLAYER_ID):
            for _ in range(200):
                pid = new_id()
                q = random_piece(pid, ctx.board_size, lens_weights, dir_bias, rnd)
                if crosses_cell(q, cx, cy):
                    tmp = pieces_copy(pieces) + [q]
                    if overlap_free(tmp, ctx.board_size):
                        pieces.append(q); cells_needed -= 1; break

    return pieces

def as_ts_level(id_str: str, pieces: List[Piece], difficulty: str, ctx: BoardContext) -> str:
    def piece_to_ts(p: Piece) -> str:
        return ("{ id: '%s', len: %d, dir: '%s', x: %d, y: %d, asset: '%s' }"
                % (p.id, p.len, p.dir, p.x, p.y, p.asset))
    pieces_ts = ",\n            ".join(piece_to_ts(p) for p in pieces)
    return f"""\
{{
    id: '{id_str}',
    size: {ctx.board_size},
    difficulty: '{difficulty}',
    exit: {{ side: '{ctx.exit_side}', index: {ctx.exit_index} }},
    pieces: [
            {pieces_ts}
    ],
}}"""

def make_levels(ctx: BoardContext,
                n_levels: int = 10,
                seed: int = 1234,
                min_moves: int = 10,
                max_moves: int = 18,
                num_pieces_range: Tuple[int, int] = (8, 12),
                cover_top_cells: int = 2,
                lens_weights: Tuple[int, int, int] = (2, 6, 2),
                dir_bias: Tuple[float, float] = (1.0, 1.0),
                difficulty_label: str = "easy",
                level_prefix: str = "e",
                max_bfs_states: int = 300000,
                max_attempts: int = 10000,
                progress_interval: int = 500,
                target_moves: Optional[int] = None) -> List[str]:
    random.seed(seed)
    rng = random.Random(seed)

    out_ts_blocks: List[str] = []
    attempts = 0
    ctx_max_states = max_bfs_states
    discard_counters = {
        "empty": 0,
        "trivial": 0,
        "unsolved": 0,
        "range": 0,
    }

    if target_moves is not None:
        min_moves = max(min_moves, target_moves - 1)
        max_moves = max(max_moves, target_moves + 2)
        # refuerza bloqueos y longitud para niveles largos
        cover_top_cells = max(cover_top_cells, 3)
        lw0, lw1, lw2 = lens_weights
        lens_weights = (lw0 + 1, max(1, lw1 - 1), lw2 + 1)
        ctx_max_states = int(ctx_max_states * 1.3)

    start_time = time.perf_counter()

    def log_progress(force: bool = False) -> None:
        if not progress_interval:
            return
        if (attempts % progress_interval != 0) and not force:
            return
        elapsed = time.perf_counter() - start_time
        success_rate = (len(out_ts_blocks) / attempts) if attempts else 0.0
        attempts_per_sec = attempts / elapsed if elapsed > 0 else 0.0
        eta = None
        if success_rate > 0 and attempts_per_sec > 0:
            remaining_levels = max(0, n_levels - len(out_ts_blocks))
            eta = (remaining_levels / success_rate) / attempts_per_sec
        progress_pct = (len(out_ts_blocks) / n_levels * 100) if n_levels else 0
        msg = [
            f"PROGRESO {progress_pct:5.1f}%", \
            f"niveles {len(out_ts_blocks)}/{n_levels}", \
            f"intentos {attempts}/{max_attempts}", \
            f"acierto {success_rate*100:4.2f}%", \
            f"t={elapsed:0.1f}s",
        ]
        if eta is not None:
            msg.append(f"eta~{eta:0.1f}s")
        msg.append(
            "descartes "
            f"vacío={discard_counters['empty']} "
            f"trivial={discard_counters['trivial']} "
            f"sin_sol={discard_counters['unsolved']} "
            f"rango={discard_counters['range']}"
        )
        sys.stderr.write(" | ".join(msg) + "\n")

    while len(out_ts_blocks) < n_levels and attempts < max_attempts:
        attempts += 1
        pieces = generate_candidate(ctx, num_pieces_range, cover_top_cells, lens_weights, dir_bias, rng=rng)
        if not pieces:
            discard_counters["empty"] += 1
            log_progress()
            continue
        if is_goal(pieces, ctx):
            discard_counters["trivial"] += 1
            log_progress()
            continue
        # solver
        res = solve_bfs(pieces, ctx, max_states=ctx_max_states)
        if res is None:
            discard_counters["unsolved"] += 1
            log_progress()
            continue
        visited, moves = res
        mcount = len(moves)
        # filtra por dificultad
        if not (min_moves <= mcount <= max_moves):
            discard_counters["range"] += 1
            log_progress()
            continue
        # ordena por id para consistencia visual
        pieces_sorted = sorted(pieces, key=lambda p: (p.id != PLAYER_ID, p.id))
        out_ts_blocks.append(
            as_ts_level(
                f"{level_prefix}{len(out_ts_blocks)+1:02d}",
                pieces_sorted,
                difficulty_label,
                ctx,
            )
        )
        log_progress()

    log_progress(force=True)
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
    ap.add_argument("--max-bfs-states", type=int, help="Límite de estados visitados por el solver")
    ap.add_argument("--max-attempts", type=int, help="Intentos máximos por lote de generación")
    ap.add_argument("--progress-interval", type=int, default=500, help="Intentos entre reportes de progreso (0=off)")
    ap.add_argument("--target-moves", type=int, help="Preset para sesgar hacia un número objetivo de movimientos")
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
    if args.max_bfs_states is not None:
        prof["max_bfs_states"] = args.max_bfs_states
    if args.max_attempts is not None:
        prof["max_attempts"] = args.max_attempts

    ctx = BoardContext(
        board_size=prof["board_size"],
        exit_side=prof["exit"]["side"],
        exit_index=prof["exit"]["index"],
        player=Piece(**prof["player"]),
    )

    blocks = make_levels(
        ctx=ctx,
        n_levels=args.n_levels,
        seed=prof["seed"],
        min_moves=prof["min_moves"],
        max_moves=prof["max_moves"],
        num_pieces_range=prof["num_pieces_range"],
        cover_top_cells=prof["cover_top_cells"],
        lens_weights=prof["lens_weights"],
        dir_bias=prof["dir_bias"],
        difficulty_label=args.difficulty,
        level_prefix=prof.get("level_prefix", args.difficulty[0]),
        max_bfs_states=prof.get("max_bfs_states", 300000),
        max_attempts=prof.get("max_attempts", 10000),
        progress_interval=args.progress_interval,
        target_moves=args.target_moves,
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
