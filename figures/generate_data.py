#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator for the numerical experiments of
"Rate-Independent Epigenetics: a thermodynamically consistent framework
 for modelling epigenetic changes".

Pure standard-library Python (only `math`, `os`). Produces .dat tables
consumed by pgfplots in main.tex. No numpy / matplotlib required.

Two benchmarks, both admitting a closed-form solution (no Cardano):

  (1) Linear play operator: convex quadratic well, single minimum.
  (2) Bistable piecewise-quadratic double well: two offset parabolas
      glued at a corner, genuinely non-convex (two minima), whose
      piecewise-LINEAR force g=W' still inverts elementarily. A
      transition between the wells under monotone loading is
      necessarily a jump; this instantiates the balanced-viscosity
      (BV) selection of Section 4 concretely, with a closed-form
      threshold and post-jump state.
"""

import math
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
#  Common loading protocol
# ==============================================================================
linf = 0.5
lam  = 1.0
Smax = 5.0
T    = 1.0

def ell(S):  return linf * (1.0 - math.exp(-lam * S))
def Sfun(t): return 0.5 * Smax * (1.0 - math.cos(2.0 * math.pi * t / T))

ELL_MAX = ell(Smax)                     # peak loading, attained at t = T/2

def write_table(fname, header, rows):
    with open(os.path.join(OUTDIR, fname), "w", encoding="utf-8") as fh:
        fh.write(header + "\n")            # pgfplots column-name row (no comment char)
        for row in rows:
            fh.write(" ".join(f"{x:.10g}" for x in row) + "\n")

def subsample(rows, target=600):
    n = len(rows)
    stride = max(1, n // target)
    out = [rows[i] for i in range(n) if i % stride == 0]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out

# ==============================================================================
#  Benchmark 1 — Linear play operator (convex quadratic well)
# ==============================================================================
#    W(q)   = (k/2) q^2                                  (convex, quadratic)
#    E(q,S) = (k/2) q^2 - q l(S)
#    f(q,S) = l(S) - k q                                 (g(q)=W'(q)=k q, linear)
#    Psi(v) = rho |v|
#
#  Setting u(t)=l(S(t))/k and r=rho/k, the state is the classical play
#  operator q=P_r[u] with q(0)=0. For a single up-down loading cycle
#  (u increasing on [0,T/2], decreasing on [T/2,T]) it is exactly:
#
#    ascending  (t<=T/2):  q(t) = max(0, u(t) - r)
#    descending (t> T/2):  q(t) = min(q_peak, u(t) + r),  q_peak = max(0, u_max - r)
#
#  Reversible vs irreversible regime is governed by rho:
#    l_max > 2 rho          -> recovery (reversible);   q_res = rho/k
#    l_max/2 < rho < l_max  -> lock-in (irreversible);   q_res = q_peak

k1 = 1.0

def W1(q):    return 0.5 * k1 * q * q
def E1(q, S): return W1(q) - q * ell(S)

def q_exact1(t, rho):
    r = rho / k1
    u = ell(Sfun(t)) / k1
    u_max = ELL_MAX / k1
    q_peak = max(0.0, u_max - r)
    if t <= 0.5 * T:
        return max(0.0, u - r)          # ascending branch
    return min(q_peak, u + r)           # descending branch

def return_map1(q_k, S_next, rho):
    f_tr = ell(S_next) - k1 * q_k
    if abs(f_tr) <= rho:
        return q_k                              # elastic lock
    if f_tr > rho:
        return (ell(S_next) - rho) / k1         # forward flow  (f = +rho)
    return (ell(S_next) + rho) / k1             # reverse flow  (f = -rho)

def run1(N, rho, q0=0.0):
    h = T / N
    ts = [i * h for i in range(N + 1)]
    Ss = [Sfun(t) for t in ts]
    qs = [q0]
    diss = work = 0.0
    for i in range(N):
        q_next = return_map1(qs[i], Ss[i + 1], rho)
        qs.append(q_next)
        diss += rho * abs(q_next - qs[i])
        work += E1(qs[i], Ss[i + 1]) - E1(qs[i], Ss[i])   # frozen-state discrete work
    defect = (E1(qs[0], Ss[0]) + work) - (E1(qs[N], Ss[N]) + diss)
    return ts, Ss, qs, defect

RHO_REV = 0.10     # reversible:   l_max > 2 rho
RHO_IRR = 0.30     # irreversible: l_max/2 < rho < l_max

for rho, tag in [(RHO_REV, "rev"), (RHO_IRR, "irr")]:
    ts, Ss, qs, _ = run1(2000, rho)
    rows_t = [(t, S, q, ell(S)) for t, S, q in zip(ts, Ss, qs)]
    rows_l = [(ell(S), q) for S, q in zip(Ss, qs)]
    write_table(f"fig_time_{tag}.dat", "t  S  q  ell", subsample(rows_t))
    write_table(f"fig_loop_{tag}.dat", "ell  q", subsample(rows_l))

rows1 = []
for N in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
    ts_h, Ss_h, qs_h, defect = run1(N, RHO_REV)
    err = max(abs(qs_h[i] - q_exact1(ts_h[i], RHO_REV)) for i in range(len(ts_h)))
    rows1.append((T / N, err, abs(defect)))
write_table("fig_convergence.dat", "h  err_state  energy_defect", rows1)

# ==============================================================================
#  Benchmark 2 — Bistable piecewise-quadratic double well
# ==============================================================================
#    W(q)   = (k/2)(q+a)^2   for q<=0,   (k/2)(q-a)^2   for q>0
#    g(q)   = k(q+a)         for q<=0,   k(q-a)         for q>0     (PIECEWISE LINEAR)
#    E(q,S) = W(q) - q l(S)
#    Psi(v) = rho |v|
#
#  Two wells at q=-a and q=+a (W=0), separated by a corner at q=0
#  (W continuous, not differentiable there: g jumps from +k*a to -k*a).
#  Starting at q(0)=-a (bottom of the left well), under increasing
#  loading the state follows the shifted play operator on the left
#  branch until the branch runs out of room; at
#
#      l_jump = k*a + rho
#
#  the (unique, BV-selected) minimiser jumps directly to
#
#      q_jump_plus = 2*a
#
#  on the right branch. Both are elementary closed forms (verified by
#  direct comparison of I*_k at q=0 vs q=2a). No Cardano anywhere.

k2   = 1.0
a2   = 0.15
rho2 = 0.10                    # l_jump = rho2 = 0.10 < ELL_MAX ~= 0.497

# The incremental scheme (eq:Ikstar) is a GLOBAL minimisation over all of Q,
# so it computes the ENERGETIC solution (Thm convergence), whose switching
# instant is set by a Maxwell-type global energy comparison, not by the
# LOCAL loss of stability of the current branch. For this symmetric
# piecewise-quadratic double well the two are algebraically distinct:
#
#   Energetic (global) jump:  l_jump = rho,        q^- = -a,  q^+ = +a
#   BV (local) jump:          l_jump = k*a + rho,   q^- =  0,  q^+ = 2a
#
# both closed form (verified by direct energy comparison, no Cardano). We
# validate the scheme against the energetic solution, since that is what
# eq:Ikstar computes; the BV threshold is quoted only for contrast with
# Section 4.
L_JUMP = rho2
Q_JUMP_MINUS = -a2
Q_JUMP_PLUS = a2
L_JUMP_BV = k2 * a2 + rho2      # local/BV threshold, for comparison only

def W2(q):
    return 0.5 * k2 * (q + a2)**2 if q <= 0.0 else 0.5 * k2 * (q - a2)**2

def g2(q):
    return k2 * (q + a2) if q <= 0.0 else k2 * (q - a2)

def E2(q, S): return W2(q) - q * ell(S)

def I_star2(q, q_k, S_next):
    """Incremental objective E(q,S_{k+1}) + rho|q-q_k|."""
    return E2(q, S_next) + rho2 * abs(q - q_k)

def return_map2(q_k, S_next):
    """Robust return map: minimise I_star2 by evaluating it at every
    candidate stationary point of each (branch, dissipation-sign)
    combination, plus the two breakpoints {0, q_k}, and taking the
    global argmin. Since I_star2 is piecewise quadratic with strictly
    convex pieces separated by these breakpoints, the true global
    minimiser is always among these candidates."""
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

# ---- exact solution, including the jump instant (closed form) -------------
S_JUMP = -math.log(1.0 - L_JUMP / linf) / lam                        # invert l(S)=l_jump
T_JUMP = (T / (2.0 * math.pi)) * math.acos(1.0 - 2.0 * S_JUMP / Smax)  # invert S(t) on [0,T/2]

Q_PEAK2 = (ELL_MAX - rho2) / k2 + a2      # right-branch state at t=T/2 (peak load)

def q_exact2(t):
    ell_t = ell(Sfun(t))
    if t <= T_JUMP:
        return Q_JUMP_MINUS                             # locked at the left well's bottom
    if t <= 0.5 * T:
        return (ell_t - rho2) / k2 + a2                 # post-jump, right branch, ascending
    return min(Q_PEAK2, (ell_t + rho2) / k2 + a2)        # descending, right branch (lock-then-reverse-flow)

Q_RES2 = q_exact2(T)

# ---- time evolution / hysteresis loop (single illustrative case) ----------
ts2, Ss2, qs2, _ = run2(2000)
rows_t2 = [(t, S, q, ell(S)) for t, S, q in zip(ts2, Ss2, qs2)]
rows_l2 = [(ell(S), q) for S, q in zip(Ss2, qs2)]
write_table("fig_time_bistable.dat", "t  S  q  ell", subsample(rows_t2))
write_table("fig_loop_bistable.dat", "ell  q", subsample(rows_l2))

# ---- convergence near the jump ---------------------------------------------
# At grid NODES the return map is exact (as in Benchmark 1): it solves the
# same threshold comparison, with the same q_k, that the exact solution
# uses, so q_h(t_i) = q_exact2(t_i) for every node. The genuine O(h)
# discretisation error is a purely INTERPOLATION effect: between the true
# jump instant T_JUMP and the next grid node, the piecewise-constant
# discrete solution still shows the pre-jump value. We therefore compare
# the piecewise-constant interpolant of q_h against q_exact2 on a FIXED,
# h-independent fine time grid (so the comparison points do not
# conveniently coincide with the marching nodes), and report: the L1
# (time-averaged) error, which sees the mismatch diluted by its shrinking
# O(h) width and is expected to scale as O(h); and the sup-norm error
# restricted to times more than a (shrinking) margin from T_JUMP, which
# isolates the "exact away from the jump" behaviour.
N_FINE = 200_000
T_FINE = [i * (T / N_FINE) for i in range(N_FINE + 1)]
Q_EXACT_FINE = [q_exact2(t) for t in T_FINE]

def interpolate_pw_constant(ts_h, qs_h, t):
    """Right-continuous piecewise-constant interpolant of the discrete
    solution: q_h(t) = qs_h[k] for t in [t_k, t_{k+1})."""
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
# Node values (t_k, q_k) for a few coarse N, to be drawn as piecewise-constant
# (const-plot) staircases, together with a fine sampling of the exact
# solution. As h shrinks the staircase refines and the jump, resolved only to
# O(h), migrates towards the exact instant T_JUMP.
N_TRAJ = [(10, "c"), (25, "m"), (100, "f")]
for N, tag in N_TRAJ:
    ts_h, Ss_h, qs_h, _ = run2(N)
    write_table(f"fig_traj_bistable_{tag}.dat", "t  q", list(zip(ts_h, qs_h)))
ts_ex = [i * (T / 1000) for i in range(1001)]
write_table("fig_traj_bistable_exact.dat", "t  q", [(t, q_exact2(t)) for t in ts_ex])
print("trajectory files: jump-node (first node with q>0) per N:")
for N, tag in N_TRAJ:
    ts_h, _, qs_h, _ = run2(N)
    jn = next((ts_h[i] for i in range(len(qs_h)) if qs_h[i] > 0.0), None)
    print(f"   N={N:4d} (h={T/N:.4f}, tag={tag}): jump appears at t={jn:.4f}  "
          f"(exact t_jump={T_JUMP:.4f})")

# ==============================================================================
#  Console summary
# ==============================================================================
def peaks1(rho):
    r = rho / k1
    q_peak = max(0.0, ELL_MAX / k1 - r)
    q_res = r if ELL_MAX > 2 * rho else q_peak
    return q_peak, q_res

print(f"[Benchmark 1] l_max = {ELL_MAX:.6f}")
for rho, name in [(RHO_REV, "reversible"), (RHO_IRR, "irreversible")]:
    qp, qr = peaks1(rho)
    _, _, qs, _ = run1(4000, rho)
    print(f"  [{name:12s}] rho={rho}: q_peak(exact)={qp:.5f} num={max(qs):.5f} | "
          f"q_res(exact)={qr:.5f} num={qs[-1]:.5f}")
print("  convergence (h, state error, energy defect):")
for h, e, d in rows1:
    print(f"     h={h:.3e}  err_state={e:.2e}  E_defect={d:.3e}")

print(f"\n[Benchmark 2] l_jump (energetic) = {L_JUMP:.6f}, "
      f"l_jump (BV, for contrast) = {L_JUMP_BV:.6f}  (< l_max = {ELL_MAX:.6f}), "
      f"t_jump = {T_JUMP:.6f}")
_, _, qs2n, _ = run2(4000)
print(f"  q_jump+ (exact) = {Q_JUMP_PLUS:.5f}  |  q_peak (exact) = {Q_PEAK2:.5f} "
      f"num_peak = {max(qs2n):.5f}")
print(f"  q_res  (exact) = {Q_RES2:.5f}  num = {qs2n[-1]:.5f}")
print("  convergence (h, L1 error, Linf away from jump, energy defect):")
for h, e1, einf, d in rows2:
    print(f"     h={h:.3e}  L1={e1:.2e}  Linf_away={einf:.2e}  E_defect={d:.3e}")
