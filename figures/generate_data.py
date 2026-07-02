#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator for the numerical experiments of
"A thermodynamically consistent framework of rate-independent evolution:
 theory and variational approximation".

Pure standard-library Python (only `math`). Produces .dat tables consumed by
pgfplots in main.tex. No numpy / matplotlib required.

Benchmark model (abstract, no biological interpretation):
    State space          Q = R
    Stored energy        W(q)   = k q^2 (1-q)^2         (symmetric double well)
    Loading (increasing) l(S)   = linf (1 - exp(-lam S))   -> bounded, Lipschitz
    Interaction energy   E(q,S) = W(q) - q l(S)
    Driving force        f(q,S) = l(S) - g(q),  g(q)=W'(q)=2k q(1-q)(1-2q)
    Dissipation          Psi(v) = rho |v|            (1-homogeneous, symmetric)

The loading l is strictly increasing and saturating, so a *high* signal S drives
q towards the second well (q -> 1). It is bounded with bounded derivative, hence
satisfies hypotheses (H_l) of the existence / convergence theorems.
"""

import math
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
#  Model parameters
# ----------------------------------------------------------------------------
k    = 1.0     # barrier scale, W(q) = k q^2 (1-q)^2   (barrier height k/16)
rho  = 0.03    # dissipation threshold (elastic half-width)
lam  = 1.0     # loading saturation constant
T    = 1.0     # final (pseudo-)time

# True fold value of g on the lower branch:  g_max = k/(3 sqrt 3)
G_MAX = k / (3.0 * math.sqrt(3.0))

def W(q):   return k * q * q * (1.0 - q) * (1.0 - q)
def g(q):   return 2.0 * k * q * (1.0 - q) * (1.0 - 2.0 * q)      # W'(q)
def gp(q):  return 2.0 * k * (6.0 * q * q - 6.0 * q + 1.0)        # W''(q)
def ell(S): return linf * (1.0 - math.exp(-lam * S))
def E(q, S): return W(q) - q * ell(S)

# smooth up-down loading over [0,T]: 0 -> Smax -> 0 (C-infinity)
def Sfun(t, Smax):
    return 0.5 * Smax * (1.0 - math.cos(2.0 * math.pi * t / T))

# ----------------------------------------------------------------------------
#  Return-map (backward-Euler incremental variational step)
# ----------------------------------------------------------------------------
def solve_branch(q_start, target, direction):
    """Solve g(q) = target moving from q_start along `direction` (+1/-1),
    via a safeguarded Newton iteration. Returns the reached root."""
    q = q_start
    for _ in range(200):
        r = g(q) - target
        d = gp(q)
        if abs(d) < 1e-14:
            q += direction * 1e-6
            continue
        step = r / d
        q_new = q - step
        # keep motion monotone in the flow direction when possible
        q = q_new
        if abs(step) < 1e-13:
            break
    return q

def return_map(q_k, S_next):
    """One incremental step: given state q_k and next load S_next, return q_{k+1}."""
    f_trial = ell(S_next) - g(q_k)
    if abs(f_trial) <= rho:
        return q_k                      # elastic lock: q_{k+1} = q_k
    if f_trial > rho:                   # forward flow (damage), f = +rho
        return solve_branch(q_k, ell(S_next) - rho, +1.0)
    else:                               # reverse flow (recovery), f = -rho
        return solve_branch(q_k, ell(S_next) + rho, -1.0)

def run(N, Smax, q0=0.0):
    """Integrate the cycle with N uniform steps. Returns trajectory + energy defect."""
    h = T / N
    ts = [i * h for i in range(N + 1)]
    Ss = [Sfun(t, Smax) for t in ts]
    qs = [q0]
    diss = 0.0
    work = 0.0
    for i in range(N):
        q_next = return_map(qs[i], Ss[i + 1])
        qs.append(q_next)
        diss += rho * abs(q_next - qs[i])
        work += E(qs[i], Ss[i + 1]) - E(qs[i], Ss[i])   # frozen-state discrete work
    # global energy-balance defect  = (E0 + work) - (E_end + diss)  >= 0
    defect = (E(qs[0], Ss[0]) + work) - (E(qs[N], Ss[N]) + diss)
    return ts, Ss, qs, defect

# ----------------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------------
def write_table(fname, header, rows):
    with open(os.path.join(OUTDIR, fname), "w", encoding="utf-8") as fh:
        fh.write(header + "\n")   # pgfplots column-name row (no comment char)
        for row in rows:
            fh.write(" ".join(f"{x:.10g}" for x in row) + "\n")

def subsample(rows, qcol, target=600, jump_tol=0.02):
    """Keep ~target rows (uniform stride) plus endpoints and any jump points,
    so pgfplots renders light figures without losing the discontinuity."""
    n = len(rows)
    stride = max(1, n // target)
    out = []
    for i, row in enumerate(rows):
        keep = (i % stride == 0) or (i == n - 1)
        if i > 0 and abs(rows[i][qcol] - rows[i - 1][qcol]) > jump_tol:
            if out and out[-1] is not rows[i - 1]:
                out.append(rows[i - 1])   # anchor before the jump
            keep = True
        if keep:
            out.append(row)
    return out

def sample_ref(ts_ref, qs_ref, t):
    """Linear interpolation of the reference trajectory at time t."""
    # ts_ref uniform on [0,T]
    N = len(ts_ref) - 1
    x = t / T * N
    i = int(math.floor(x))
    if i >= N:
        return qs_ref[N]
    frac = x - i
    return qs_ref[i] * (1.0 - frac) + qs_ref[i + 1] * frac

# ----------------------------------------------------------------------------
#  Experiment 1 : smooth reversible cycle  (l_inf - rho < g_max)
# ----------------------------------------------------------------------------
linf = 0.18
Smax = 5.0
assert linf - rho < G_MAX, "params should give a smooth (jump-free) loop"

ts, Ss, qs, _ = run(2000, Smax)
rows_time = [(t, S, q, ell(S)) for t, S, q in zip(ts, Ss, qs)]
rows_loop = [(ell(S), q, S) for S, q in zip(Ss, qs)]
write_table("fig_time_smooth.dat", "t  S  q  ell", subsample(rows_time, qcol=2))
write_table("fig_loop_smooth.dat", "ell  q  S", subsample(rows_loop, qcol=1))

qmax_s = max(qs)
qres_s = qs[-1]

# ----------------------------------------------------------------------------
#  Experiment 2 : catastrophic (EMT-like) cycle  (l_inf - rho > g_max)
# ----------------------------------------------------------------------------
linf = 0.30
assert linf - rho > G_MAX, "params should trigger a forward catastrophe"
ts2, Ss2, qs2, _ = run(4000, Smax)
rows_loop_c = [(ell(S), q, S) for S, q in zip(Ss2, qs2)]
rows_time_c = [(t, S, q, ell(S)) for t, S, q in zip(ts2, Ss2, qs2)]
write_table("fig_loop_cat.dat", "ell  q  S", subsample(rows_loop_c, qcol=1))
write_table("fig_time_cat.dat", "t  S  q  ell", subsample(rows_time_c, qcol=2))
qmax_c = max(qs2)
qres_c = qs2[-1]

# ----------------------------------------------------------------------------
#  Experiment 3 : convergence study on the smooth cycle
# ----------------------------------------------------------------------------
linf = 0.18                                   # back to the smooth case
ts_ref, Ss_ref, qs_ref, _ = run(200000, Smax)  # fine reference

rows = []
for N in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
    ts_h, Ss_h, qs_h, defect = run(N, Smax)
    err_inf = max(abs(qs_h[i] - sample_ref(ts_ref, qs_ref, ts_h[i]))
                  for i in range(len(ts_h)))
    h = T / N
    rows.append((h, err_inf, abs(defect)))
write_table("fig_convergence.dat", "h  err_inf  energy_defect", rows)

# ----------------------------------------------------------------------------
#  Experiment 4 : energy landscape at three load levels (illustrative)
# ----------------------------------------------------------------------------
linf = 0.18
grid = [(-0.20 + 1.40 * j / 400.0) for j in range(401)]
for Sval, tag in [(0.0, "lo"), (1.0, "mid"), (5.0, "hi")]:
    write_table(f"fig_landscape_{tag}.dat", "q  E",
                [(q, E(q, Sval)) for q in grid])

# ----------------------------------------------------------------------------
#  Console summary
# ----------------------------------------------------------------------------
print(f"g_max = k/(3 sqrt 3) = {G_MAX:.6f}")
print(f"[smooth]  l_inf-rho = {0.18-rho:.4f} < g_max -> reversible")
print(f"          q_max = {qmax_s:.5f}  q_res = {qres_s:.5f}")
print(f"[catastro] l_inf-rho = {0.30-rho:.4f} > g_max -> jump")
print(f"          q_max = {qmax_c:.5f}  q_res = {qres_c:.5f}")
print("convergence (h, err_inf, energy_defect):")
for r in rows:
    print(f"   h={r[0]:.3e}   err={r[1]:.3e}   Edef={r[2]:.3e}")
print("Wrote: fig_time_smooth, fig_loop_smooth, fig_loop_cat, fig_time_cat,")
print("       fig_convergence, fig_landscape_{lo,mid,hi}.dat")
