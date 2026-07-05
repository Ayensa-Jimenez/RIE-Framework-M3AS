#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark 3 for "Rate-Independent Epigenetics" (Paper 1): a NONLINEAR
convex benchmark validated by the method of manufactured solutions.

Purpose (addresses reviewer point R1): the first two benchmarks have a
force g = W' that is linear (play operator) or piecewise linear (bilinear
double well), so the scalar return map inverts g in closed form and never
needs an iterative solve. Here g is genuinely transcendental with no
elementary inverse, so the return map MUST solve g(q) = target by a
generic (safeguarded Newton) iteration. We show that the scheme retains
node-exactness and O(h) global behaviour in this genuinely nonlinear
setting, i.e. the method does not rely on closed-form invertibility.

Model (scalar, convex):
    W(q)   = q^2/2 + cosh(q) - 1          (uniformly convex: W'' = 1+cosh q >= 2)
    g(q)   = W'(q) = q + sinh(q)          (transcendental, no elementary inverse)
    Psi(v) = rho |v|
    E(q,t) = W(q) - q*ell(t),   f = ell - g(q)

Manufactured solution: we PRESCRIBE a smooth piecewise-monotone q(t)
(lock -> forward flow -> lock -> backward flow -> lock) and DERIVE the
loading ell(t) that makes it the exact play-operator response, using only
FORWARD evaluations of g (never g^{-1}):

    forward flow  (qdot>0):  ell(t) = g(q(t)) + rho
    backward flow (qdot<0):  ell(t) = g(q(t)) - rho
    lock         (qdot=0):   ell(t) sweeps inside the elastic band [g(q)-rho, g(q)+rho]

The numerical return map, by contrast, is given only ell at the nodes and
must recover q by solving g(q)=ell -/+ rho with Newton.

Pure standard-library Python (math only); no numpy / matplotlib.
"""

import math
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
#  Model
# ----------------------------------------------------------------------------
rho = 0.3
T   = 1.0

def W(q):  return 0.5 * q * q + math.cosh(q) - 1.0
def g(q):  return q + math.sinh(q)          # transcendental force
def gp(q): return 1.0 + math.cosh(q)        # g'(q) >= 2 > 0  (for Newton)
def E(q, ell): return W(q) - q * ell

# ----------------------------------------------------------------------------
#  Manufactured exact solution q_exact(t) and the loading ell(t) it induces
# ----------------------------------------------------------------------------
TA, TB, TC, TD = 0.15, 0.45, 0.55, 0.85     # phase boundaries
QMIN, QMAX = 0.0, 1.0

def smoothstep(x):
    """C^1 smoothstep on [0,1]: s(0)=0, s(1)=1, s'(0)=s'(1)=0."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    return x * x * (3.0 - 2.0 * x)

def q_exact(t):
    if t <= TA:                              # initial lock
        return QMIN
    if t <= TB:                              # forward flow
        return QMIN + (QMAX - QMIN) * smoothstep((t - TA) / (TB - TA))
    if t <= TC:                              # lock at peak
        return QMAX
    if t <= TD:                              # backward flow
        return QMAX + (QMIN - QMAX) * smoothstep((t - TC) / (TD - TC))
    return QMIN                              # final lock

def ell_exact(t):
    """Loading that makes q_exact the exact play-operator response.
    Uses only forward evaluations of g."""
    if t <= TA:                              # lock: rise g(QMIN) -> g(QMIN)+rho
        return g(QMIN) + rho * (t / TA)
    if t <= TB:                              # forward flow: ell = g(q) + rho
        return g(q_exact(t)) + rho
    if t <= TC:                              # lock at peak: g(QMAX)+rho -> g(QMAX)-rho
        s = (t - TB) / (TC - TB)
        return g(QMAX) + rho * (1.0 - 2.0 * s)
    if t <= TD:                              # backward flow: ell = g(q) - rho
        return g(q_exact(t)) - rho
    s = (t - TD) / (T - TD)                  # final lock: g(QMIN)-rho -> g(QMIN)
    return g(QMIN) - rho + rho * s

TRANSITIONS = (TA, TB, TC, TD)

# ----------------------------------------------------------------------------
#  Generic safeguarded Newton for g(q) = target  (NO closed-form inverse)
# ----------------------------------------------------------------------------
def solve_g(target, q0, tol=1e-13, maxit=100):
    """Solve g(q)=target. Newton (g'>=2>0 => well-conditioned), safeguarded
    by bisection on a bracket grown from the initial guess. Returns
    (q, iters, final_residual)."""
    # grow a bracket [lo,hi] around q0 with g(lo)<=target<=g(hi)
    lo, hi = q0, q0
    step = 1.0
    while g(lo) > target:
        lo -= step; step *= 2.0
    step = 1.0
    while g(hi) < target:
        hi += step; step *= 2.0
    q = q0
    if not (lo <= q <= hi):
        q = 0.5 * (lo + hi)
    it = 0
    for it in range(1, maxit + 1):
        r = g(q) - target
        if abs(r) < tol:
            return q, it, abs(r)
        # bracket update
        if r > 0.0: hi = q
        else:       lo = q
        qn = q - r / gp(q)                    # Newton step
        if not (lo < qn < hi):               # safeguard: fall back to bisection
            qn = 0.5 * (lo + hi)
        q = qn
    return q, it, abs(g(q) - target)

def return_map(q_k, ell_next):
    """Incremental update. Returns (q_next, newton_iters, residual)."""
    f_trial = ell_next - g(q_k)
    if abs(f_trial) <= rho:                  # elastic lock
        return q_k, 0, 0.0
    target = ell_next - math.copysign(rho, f_trial)   # yield surface
    return solve_g(target, q_k)              # generic solve (Newton)

# ----------------------------------------------------------------------------
#  Time stepping
# ----------------------------------------------------------------------------
def run(N):
    h = T / N
    ts = [i * h for i in range(N + 1)]
    qs = [q_exact(0.0)]
    diss = work = 0.0
    newton_iters_total = 0
    flow_steps = 0
    max_iters = 0
    max_resid = 0.0
    for i in range(N):
        ell_next = ell_exact(ts[i + 1])
        q_next, it, res = return_map(qs[i], ell_next)
        qs.append(q_next)
        diss += rho * abs(q_next - qs[i])
        work += E(qs[i], ell_next) - E(qs[i], ell_exact(ts[i]))
        if it > 0:
            newton_iters_total += it
            flow_steps += 1
            max_iters = max(max_iters, it)
            max_resid = max(max_resid, res)
    defect = (E(qs[0], ell_exact(ts[0])) + work) - (E(qs[N], ell_exact(ts[N])) + diss)
    avg_it = newton_iters_total / flow_steps if flow_steps else 0.0
    return ts, qs, defect, avg_it, max_iters, max_resid

# ----------------------------------------------------------------------------
#  Output helpers
# ----------------------------------------------------------------------------
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
#  Self-checks (printed, not silent) before producing the dataset
# ----------------------------------------------------------------------------
print("=== Benchmark 3: nonlinear convex, manufactured solution ===")
print(f"g(q)=q+sinh(q), W(q)=q^2/2+cosh(q)-1, rho={rho}")

# (a) convexity of W on the working range
qs_grid = [-1.5 + 3.0 * j / 400 for j in range(401)]
min_Wpp = min(gp(q) for q in qs_grid)   # W'' = g'
print(f"[convexity]   min W''(q) on [-1.5,1.5] = {min_Wpp:.4f}  (>0 required)")

# (b) continuity of ell at the four phase transitions
print("[ell continuity] jump magnitudes at transitions:")
for tb_ in TRANSITIONS:
    left  = ell_exact(tb_ - 1e-9)
    right = ell_exact(tb_ + 1e-9)
    print(f"    t={tb_:.2f}:  |ell(t-)-ell(t+)| = {abs(left-right):.2e}")

# (c) admissibility: |f|<=rho on lock, |f|=rho on flow (sampled)
def phase(t):
    if t < TA or (TC <= t < TD) is False and t > TD: pass
    return None
worst_lock = 0.0
worst_flow = 0.0
for j in range(1, 2000):
    t = T * j / 2000
    f = ell_exact(t) - g(q_exact(t))
    in_flow = (TA < t < TB) or (TC < t < TD)
    if in_flow:
        worst_flow = max(worst_flow, abs(abs(f) - rho))
    else:
        worst_lock = max(worst_lock, max(0.0, abs(f) - rho))
print(f"[admissibility] max ||f|-rho| on flow = {worst_flow:.2e}   "
      f"max (|f|-rho)+ on lock = {worst_lock:.2e}")

# (d) node-exactness and Newton behaviour at a fine grid
ts, qs, defect, avg_it, max_it, max_res = run(4000)
node_err = max(abs(qs[i] - q_exact(ts[i])) for i in range(len(ts)))
print(f"[node-exact]  max_i |q^h(t_i)-q_exact(t_i)| (N=4000) = {node_err:.2e}")
print(f"[newton]      avg iters/flow-step = {avg_it:.2f}, max iters = {max_it}, "
      f"max residual = {max_res:.2e}")

# ----------------------------------------------------------------------------
#  Datasets
# ----------------------------------------------------------------------------
# time evolution + loading (numerical run, fine)
ts, qs, *_ = run(2000)
rows_t = [(t, q, ell_exact(t)) for t, q in zip(ts, qs)]
rows_l = [(ell_exact(t), q) for t, q in zip(ts, qs)]
write_table("fig_time_nonlinear.dat", "t  q  ell", subsample(rows_t))
write_table("fig_loop_nonlinear.dat", "ell  q", subsample(rows_l))

# trajectories at a few step sizes (const-plot staircases) + exact
for N, tag in [(12, "c"), (30, "m"), (120, "f")]:
    ts_h, qs_h, *_ = run(N)
    write_table(f"fig_traj_nl_{tag}.dat", "t  q", list(zip(ts_h, qs_h)))
ts_ex = [i * (T / 1000) for i in range(1001)]
write_table("fig_traj_nl_exact.dat", "t  q", [(t, q_exact(t)) for t in ts_ex])

# convergence study
N_FINE = 200_000
T_FINE = [i * (T / N_FINE) for i in range(N_FINE + 1)]
Q_EX_FINE = [q_exact(t) for t in T_FINE]

def pw_const(ts_h, qs_h, t):
    N = len(ts_h) - 1
    idx = int(t / (T / N))
    return qs_h[min(idx, N)]

rows_conv = []
print("convergence (h, L1, Linf_away, energy_defect, avg_newton):")
for N in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
    ts_h, qs_h, defect, avg_it, _, _ = run(N)
    h = T / N
    errs = [abs(pw_const(ts_h, qs_h, t) - qe) for t, qe in zip(T_FINE, Q_EX_FINE)]
    err_l1 = sum(errs) * (T / N_FINE) / T
    margin = max(2 * h, 3 * (T / N_FINE))
    err_linf_away = max(e for e, t in zip(errs, T_FINE)
                        if all(abs(t - tb_) > margin for tb_ in TRANSITIONS))
    rows_conv.append((h, err_l1, err_linf_away, abs(defect), avg_it))
    print(f"   h={h:.3e}  L1={err_l1:.2e}  Linf_away={err_linf_away:.2e}  "
          f"E_def={abs(defect):.2e}  newton={avg_it:.2f}")
write_table("fig_convergence_nonlinear.dat",
            "h  err_L1  err_Linf_away  energy_defect  avg_newton", rows_conv)

print("--- .dat files written ---")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith(".dat") and ("nonlinear" in f or "_nl_" in f):
        print("   " + f)
