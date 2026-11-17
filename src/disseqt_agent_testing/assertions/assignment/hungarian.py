"""Maximum-weight assignment on a rectangular matrix (original code; no dependency).

The classic shortest-augmenting-path Hungarian algorithm with potentials, O(n^2 m). Weights may
be any finite floats; the caller drops zero-weight pairs afterwards.
"""

from collections.abc import Sequence


def assign_max(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return sorted ``(row, col)`` pairs maximizing the total weight (every row of the smaller
    side is assigned; pairs may carry zero weight)."""
    n = len(weights)
    m = len(weights[0]) if n else 0
    if n == 0 or m == 0:
        return []
    transposed = n > m
    if transposed:
        cost = [[-weights[i][j] for i in range(n)] for j in range(m)]
        rows, cols = m, n
    else:
        cost = [[-weights[i][j] for j in range(m)] for i in range(n)]
        rows, cols = n, m
    inf = float("inf")
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)  # p[j] = row assigned to column j (1-based), 0 = free
    way = [0] * (cols + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, cols + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)
