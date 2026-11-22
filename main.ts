// Push-optimal Sokoban solver, room by room.
// One program, no dependencies, no network, no clock, no randomness: the same
// input always writes the same bytes. Written by hand in this session.
// Method: split the level into rooms (the hallway's dead squares cut the board),
// solve each room push-optimally with A* over pushes (min-cost matching
// heuristic, live-square + frozen-2x2 pruning), then stitch the rooms together
// with the player's walks. A box never leaves its room, so the rooms' optima add
// up to the level's push optimum. If the split fails, fall back to A* over the
// whole level; if that fails too, answer empty (0 points for that instance, but
// the other instances still score and the run stays valid).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// dx, dy, walk letter, push letter. Fixed order everywhere: deterministic.
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];
const ROOM_STATES = 250_000;
const WHOLE_STATES = 500_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box, pushed alone, can still reach some goal: pull back
// from the goals. A box off a live square is dead, whatever the other boxes do.
function liveSquares(L: Level): Uint8Array {
  const { w, wall, goals } = L;
  const live = new Uint8Array(wall.length);
  const q: number[] = [...goals];
  for (const g of goals) live[g] = 1;
  for (let i = 0; i < q.length; i++) {
    const x = q[i]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length || y - d < 0 || y - d >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        q.push(y);
      }
    }
  }
  return live;
}

// Shortest walk distances from `from` over open floor (boxes ignored), -1 if cut off.
function distances(L: Level, from: number): Int32Array {
  const dist = new Int32Array(L.wall.length).fill(-1);
  const q: number[] = [from];
  dist[from] = 0;
  for (let i = 0; i < q.length; i++) {
    const p = q[i]!;
    for (const [dx, dy] of DIRS) {
      const n = p + dy * L.w + dx;
      if (n < 0 || n >= L.wall.length || L.wall[n] || dist[n] >= 0) continue;
      dist[n] = dist[p]! + 1;
      q.push(n);
    }
  }
  return dist;
}

// The player's reachable region from `from` with `occ` in the way: its lowest square.
function regionLow(L: Level, occ: Uint8Array, from: number): number {
  const seen = new Uint8Array(L.wall.length);
  const stack: number[] = [from];
  seen[from] = 1;
  let low = from;
  while (stack.length > 0) {
    const p = stack.pop()!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const n = p + dy * L.w + dx;
      if (n < 0 || n >= L.wall.length || L.wall[n] || occ[n] || seen[n]) continue;
      seen[n] = 1;
      stack.push(n);
    }
  }
  return low;
}

// Whether the square set the player reaches from `from` holds `to`.
function reaches(L: Level, occ: Uint8Array, from: number, to: number): boolean {
  if (from === to) return true;
  const seen = new Uint8Array(L.wall.length);
  const stack: number[] = [from];
  seen[from] = 1;
  while (stack.length > 0) {
    const p = stack.pop()!;
    for (const [dx, dy] of DIRS) {
      const n = p + dy * L.w + dx;
      if (n < 0 || n >= L.wall.length || L.wall[n] || occ[n] || seen[n]) continue;
      if (n === to) return true;
      seen[n] = 1;
      stack.push(n);
    }
  }
  return false;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and
// boxes holding a box off its goal: no push ever frees it.
function frozenBlock(L: Level, occ: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = L;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes, lower-case letters.
function walkTo(L: Level, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(L.wall.length).fill(-1);
  const how: string[] = new Array(L.wall.length);
  const q: number[] = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const p = q[i]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const n = p + dy * L.w + dx;
      if (n < 0 || n >= L.wall.length || L.wall[n] || occ[n] || prev[n] >= 0) continue;
      prev[n] = p;
      how[n] = letter;
      q.push(n);
    }
  }
  if (prev[to] < 0) throw new Error('walk target out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

// Push-optimal solution of one (sub-)level: {pushes, moves, endPlayer} or null.
// A* over pushes. A state is the boxes plus the player's reachable region
// (named by its lowest square). The heuristic is the cheapest assignment of
// boxes to goals by walk distance (exact for few boxes, sum-of-minima for many):
// each push moves one box one square, so it never overestimates.
function solvePush(L: Level, maxStates: number): { pushes: number; moves: string; endPlayer: number } | null {
  const n = L.boxes.length;
  const size = L.wall.length;
  const goal = new Uint8Array(size);
  for (const g of L.goals) goal[g] = 1;
  const live = liveSquares(L);
  if (L.boxes.some((b) => !live[b])) return null;
  const distTo = L.goals.map((g) => distances(L, g));
  const exact = n <= 6;
  const perms: number[][] = [];
  if (exact) {
    const idx = L.goals.map((_, i) => i);
    const gen = (a: number[], k: number): void => {
      if (k === 1) {
        perms.push(a.slice());
        return;
      }
      for (let i = 0; i < k; i++) {
        gen(a, k - 1);
        const j = k % 2 === 0 ? i : 0;
        [a[j], a[k - 1]] = [a[k - 1]!, a[j]!];
      }
    };
    gen(idx, idx.length);
  }
  const heuristic = (boxes: number[]): number => {
    if (exact) {
      let best = Infinity;
      for (const p of perms) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          const d = distTo[p[i]!]![boxes[i]!]!;
          if (d < 0) {
            s = Infinity;
            break;
          }
          s += d;
        }
        if (s < best) best = s;
      }
      return best;
    }
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (const d of distTo) if (d[b]! >= 0 && d[b]! < m) m = d[b]!;
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };

  const occ = new Uint8Array(size);
  const keyOf = (boxes: number[], low: number): string => low + '|' + boxes.join(',');
  const boxesOf: number[][] = [L.boxes.slice()];
  const playerOf: number[] = [L.player];
  const parent: number[] = [-1];
  const fromOf: number[] = [-1];
  const toOf: number[] = [-1];
  const gOf: number[] = [0];
  const seen = new Map<string, number>();
  for (const b of L.boxes) occ[b] = 1;
  seen.set(keyOf(L.boxes, regionLow(L, occ, L.player)), 0);
  for (const b of L.boxes) occ[b] = 0;
  // Open heap on [f, h, state]: f, then h, then insertion order (the state index).
  const heap: [number, number, number][] = [];
  const h0 = heuristic(L.boxes);
  if (h0 === Infinity) return null;
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, h0, 0]);
  let found = L.boxes.every((b) => goal[b]) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const [, , s] = heapPop();
    const g = gOf[s]!;
    const boxes = boxesOf[s]!;
    for (const b of boxes) occ[b] = 1;
    const low = regionLow(L, occ, playerOf[s]!);
    for (let bi = 0; bi < n && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy, , pushLetter] of DIRS) {
        const d = dy * L.w + dx;
        const t = b + d;
        const behind = b - d;
        if (t < 0 || t >= size || behind < 0 || behind >= size) continue;
        if (L.wall[t] || occ[t] || !live[t]) continue;
        if (!reaches(L, occ, playerOf[s]!, behind)) continue;
        occ[b] = 0;
        occ[t] = 1;
        const frozen = frozenBlock(L, occ, goal, t);
        let key = '';
        let h = Infinity;
        let next: number[] = [];
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          h = heuristic(next);
          if (h < Infinity) key = keyOf(next, regionLow(L, occ, b));
        }
        occ[t] = 0;
        occ[b] = 1;
        if (frozen || h === Infinity) continue;
        const old = seen.get(key);
        if (old !== undefined && gOf[old]! <= g + 1) continue;
        if (boxesOf.length >= maxStates) {
          for (const x of boxes) occ[x] = 0;
          return null;
        }
        const ns = boxesOf.length;
        seen.set(key, ns);
        boxesOf.push(next);
        playerOf.push(b);
        parent.push(s);
        fromOf.push(b);
        toOf.push(t);
        gOf.push(g + 1);
        if (next.every((x) => goal[x])) {
          found = ns;
          break;
        }
        heapPush([g + 1 + h, h, ns]);
        void pushLetter;
      }
    }
    for (const b of boxes) occ[b] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = L.player;
  let boxes = L.boxes.slice();
  for (const s of path) {
    const next = boxesOf[s]!;
    const from = fromOf[s]!;
    const to = toOf[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * L.w + dx === d)!;
    for (const b of boxes) occ[b] = 1;
    moves += walkTo(L, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    moves += dir[3];
    player = from;
    boxes = next;
  }
  return { pushes: path.length, moves, endPlayer: player };
}

// The level's rooms: cut the hallway row's dead squares out of the floor; each
// part left that holds boxes is a room. Null when the level is no set of rooms.
function roomsOf(L: Level): { boxes: number[]; goals: number[]; done: boolean }[] | null {
  const live = liveSquares(L);
  const row = Math.floor(L.player / L.w);
  const cut = new Uint8Array(L.wall.length);
  for (let x = 0; x < L.w; x++) {
    const i = row * L.w + x;
    if (!L.wall[i] && !live[i]) cut[i] = 1;
  }
  if (L.boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(L.wall.length).fill(-1);
  const rooms: { boxes: number[]; goals: number[]; done: boolean }[] = [];
  for (let s = 0; s < L.wall.length; s++) {
    if (L.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * L.w + dx;
        if (q < 0 || q >= L.wall.length || L.wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = rooms.length;
        squares.push(q);
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of L.boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of L.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function solveLevel(text: string): string {
  const L = parseLevel(text);
  const rooms = roomsOf(L);
  if (rooms === null) {
    const whole = solvePush(L, WHOLE_STATES);
    return whole === null ? '' : whole.moves;
  }
  let moves = '';
  let player = L.player;
  for (const room of rooms) {
    const wall = L.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: L.w,
      h: L.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePush(sub, ROOM_STATES);
    if (solved === null) {
      const whole = solvePush(L, WHOLE_STATES);
      return whole === null ? '' : whole.moves;
    }
    moves += solved.moves;
    player = solved.endPlayer;
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
