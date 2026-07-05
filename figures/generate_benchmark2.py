#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark 2 data generator for "Rate-Independent Epigenetics".

The BISTABLE piecewise-quadratic DOUBLE WELL: two offset parabolas glued
at a corner, genuinely non-convex (two minima), whose piecewise-LINEAR
force g=W' still inverts elementarily. A transition between the wells
under monotone loading is necessarily a jump; this instantiates the
balanced-viscosity (BV) selection of Section 4 concretely, with a
closed-form threshold and post-jump state.

    W(q)   = (k/2)(q+a)^2   for q<=0,   (k/2)(q-a)^2   for q>0
    g(q)   = k(q+a)         for q<=0,   k(q-a)         for q>0     (PIECEWISE LINEAR)
    E(q,S) = W(q) - q l(S)
    Psi(v) = rho |v|

The incremental scheme minimises GLOBALLY over Q, hence computes the
ENERGETIC solution, whose switch is set by a Maxwell-type global energy
comparison (l_jump=rho, q^-=-a -> q^+=+a), NOT by the local loss of
stability of the current branch (l_jump=k*a+rho, quoted for contrast).

Pure standard-library Python (math, os only). No numpy / matplotlib.
Writes the pgfplots .dat tables consumed by sections/experiments.tex.
"""

import math
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
#  Common loading protocol
# ----------------------------------------------------------------------------
linf = 0.5
lam  = 1.0
Smax = 5.0
T    = 1.0

def ell(S):  return linf * (1.0 - math.exp(-lam * S))
def Sfun(t): return 0.5 * Smax * (1.0 - math.cos(2.0 * math.pi * t / T))

ELL_MAX = ell(Smax)

def write_table(fname, header, rows):
    with open(os.path.join(OUTDIR, fname), "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for row in rows:
            fh.write(" ".join(f"{x:.10g}" for x in row) + "\n")

def subsample(rows, target=600):
    n = len(rows)
    stride = max(1, n // target)
    out = [rows[i] for i in range(n) if i % stride == 0]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out

# ----------------------------------------------------------------------------
#  Model
# ----------------------------------------------------------------------------
k2   = 1.0
a2   = 0.15
rho2 = 0.10                    # l_jump = rho2 = 0.10 < ELL_MAX ~= 0.497

L_JUMP = rho2
Q_JUMP_MINUS = -a2
Q_JUMP_PLUS = a2
L_JUMP_BV = k2 * a2 + rho2      # local/BV threshold, for comparison only

def W2(q):
    return 0.5 * k2 * (q + a2)**2 if q <= 0.0 else 0.5 * k2 * (q - a2)**2

def E2(q, S): return W2(q) - q * ell(S)

def I_star2(q, q_k, S_next):
    """Incremental objective E(q,S_{k+1}) + rho|q-q_k|."""
    return E2(q, S_next) + rho2 * abs(q - q_k)

def return_map2(q_k, S_next):
    """Robust return map: minimise I_star2 by evaluating it at every
    candidate stationary point of each (branch, dissipation-sign)
    combination, plus the two breakpoints {0, q_k}, and taking the global
    argmin. Since I_star2 is piecewise quadratic with strictly convex
    pieces separated by these breakpoints, the true global minimiser is
    always among these candidates."""
    ell_n = ell(S_next)
    candidates = {q_k, 0.0}
    for sign in (+1.0, -1.0):                 # dissipation branch: +rho if q>q_k, -rho if q<q_k
        candidates.add((ell_n - sign * rho2) / k2 - a2)   # stationary point, left branch (q<=0)
        candidates.add((ell_n - sign * rho2) / k2 + a2)   # stationary point, right branch (q>0)
    return min(candidates, key=lambda q: I_star2(q, q_k, S_next))

def run2(N, q0=None):
    if q0 is None:
        q0 = -a2
    h = T / N
    ts = [i * h for i in range(N + 1)]
    Ss = [Sfun(t) for t in ts]
    qs = [q0]
    diss = work = 0.0
    for i in range(N):
        q_next = return_map2(qs[i], Ss[i + 1])
        qs.append(q_next)
        diss += rho2 * abs(q_next - qs[i])
        work += E2(qs[i], Ss[i + 1]) - E2(qs[i], Ss[i])
    defect = (E2(qs[0], Ss[0]) + work) - (E2(qs[N], Ss[N]) + diss)
    return ts, Ss, qs, defect

# ---- exact solution, including the jump instant (closed form) --------------
S_JUMP = -math.log(1.0 - L_JUMP / linf) / lam                        # invert l(S)=l_jump
T_JUMP = (T / (2.0 * math.pi)) * math.acos(1.0 - 2.0 * S_JUMP / Smax)  # invert S(t) on [0,T/2]
Q_PEAK2 = (ELL_MAX - rho2) / k2 + a2      # right-branch state at t=T/2 (peak load)

def q_exact2(t):
    ell_t = ell(Sfun(t))
    if t <= T_JUMP:
        return Q_JUMP_MINUS                             # locked at the left well's bottom
    if t <= 0.5 * T:
        return (ell_t - rho2) / k2 + a2                 # post-jump, right branch, ascending
    return min(Q_PEAK2, (ell_t + rho2) / k2 + a2)        # descending, right branch

Q_RES2 = q_exact2(T)

# ---- time evolution / hysteresis loop --------------------------------------
ts2, Ss2, qs2, _ = run2(2000)
rows_t2 = [(t, S, q, ell(S)) for t, S, q in zip(ts2, Ss2, qs2)]
rows_l2 = [(ell(S), q) for S, q in zip(Ss2, qs2)]
write_table("fig_time_bistable.dat", "t  S  q  ell", subsample(rows_t2))
write_table("fig_loop_bistable.dat", "ell  q", subsample(rows_l2))

# ---- convergence near the jump ---------------------------------------------
# At grid NODES the return map is exact: it solves the same threshold
# comparison, with the same q_k, that the exact solution uses, so
# q_h(t_i) = q_exact2(t_i) at every node. The genuine O(h) error is a pure
# INTERPOLATION effect: between T_JUMP and the next node the piecewise-
# constant q_h still shows the pre-jump value. We compare the piecewise-
# constant interpolant against q_exact2 on a FIXED fine grid and report the
# L1 error and the sup error away from a shrinking margin around T_JUMP.
N_FINE = 200_000
T_FINE = [i * (T / N_FINE) for i in range(N_FINE + 1)]
Q_EXACT_FINE = [q_exact2(t) for t in T_FINE]

def interpolate_pw_constant(ts_h, qs_h, t):
    N = len(ts_h) - 1
    idx = int(t / (T / N))
    if idx >= N:
        idx = N
    return qs_h[idx]

rows2 = []
for N in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
    ts_h, Ss_h, qs_h, defect = run2(N)
    h = T / N
    errs = [abs(interpolate_pw_constant(ts_h, qs_h, t) - qe)
            for t, qe in zip(T_FINE, Q_EXACT_FINE)]
    err_l1 = sum(errs) * (T / N_FINE) / T
    margin = max(2 * h, 3 * (T / N_FINE))
    err_linf_away = max(e for e, t in zip(errs, T_FINE) if abs(t - T_JUMP) > margin)
    rows2.append((h, err_l1, err_linf_away, abs(defect)))
write_table("fig_convergence_bistable.dat",
            "h  err_L1  err_Linf_away  energy_defect", rows2)

# ---- trajectories at a few step sizes (visual convergence) -----------------
N_TRAJ = [(10, "c"), (25, "m"), (100, "f")]
for N, tag in N_TRAJ:
    ts_h, Ss_h, qs_h, _ = run2(N)
    write_table(f"fig_traj_bistable_{tag}.dat", "t  q", list(zip(ts_h, qs_h)))
ts_ex = [i * (T / 1000) for i in range(1001)]
write_table("fig_traj_bistable_exact.dat", "t  q", [(t, q_exact2(t)) for t in ts_ex])

# ----------------------------------------------------------------------------
#  Console summary
# ----------------------------------------------------------------------------
print(f"[Benchmark 2] l_jump (energetic) = {L_JUMP:.6f}, "
      f"l_jump (BV, for contrast) = {L_JUMP_BV:.6f}  (< l_max = {ELL_MAX:.6f}), "
      f"t_jump = {T_JUMP:.6f}")
_, _, qs2n, _ = run2(4000)
print(f"  q_jump+ (exact) = {Q_JUMP_PLUS:.5f}  |  q_peak (exact) = {Q_PEAK2:.5f} "
      f"num_peak = {max(qs2n):.5f}")
print(f"  q_res  (exact) = {Q_RES2:.5f}  num = {qs2n[-1]:.5f}")
print("  jump-node (first node with q>0) per coarse N:")
for N, tag in N_TRAJ:
    ts_h, _, qs_h, _ = run2(N)
    jn = next((ts_h[i] for i in range(len(qs_h)) if qs_h[i] > 0.0), None)
    print(f"     N={N:4d} (h={T/N:.4f}): jump at t={jn:.4f}  (exact {T_JUMP:.4f})")
print("  convergence (h, L1 error, Linf away from jump, energy defect):")
for h, e1, einf, d in rows2:
    print(f"     h={h:.3e}  L1={e1:.2e}  Linf_away={einf:.2e}  E_defect={d:.3e}")
