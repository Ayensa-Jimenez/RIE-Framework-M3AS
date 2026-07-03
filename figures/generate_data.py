#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator for the numerical experiments of
"Rate-Independent Epigenetics: a thermodynamically consistent framework
 for modelling epigenetic changes".

Pure standard-library Python (only `math`, `os`). Produces .dat tables
consumed by pgfplots in main.tex. No numpy / matplotlib required.

Benchmark: the linear play operator (convex quadratic well), for which the
evolution q(t) is available in closed form (no Cardano), so the return-map
integrator can be validated against the exact solution.

    State space      Q = R
    Stored energy    W(q)   = (k/2) q^2                 (convex, quadratic)
    Loading          l(S)   = linf (1 - exp(-lam S))    (increasing, bounded, Lipschitz)
    Energy           E(q,S) = (k/2) q^2 - q l(S)
    Driving force    f(q,S) = l(S) - k q               (g(q)=W'(q)=k q, linear)
    Dissipation      Psi(v) = rho |v|

Setting u(t)=l(S(t))/k and r=rho/k, the state is the classical play
operator q=P_r[u] with q(0)=0.  For a single up-down loading cycle
(u increasing on [0,T/2], decreasing on [T/2,T]) it is exactly:

    ascending  (t<=T/2):  q(t) = max(0, u(t) - r)
    descending (t> T/2):  q(t) = min(q_peak, u(t) + r),  q_peak = max(0, u_max - r)

Reversible vs irreversible regime is governed by rho:
    l_max > 2 rho  ->  recovery (reversible);  q_res = rho/k
    l_max/2 < rho < l_max  ->  lock-in (irreversible);  q_res = q_peak
"""

import math
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
#  Model parameters (fixed loading; the regime is selected by rho)
# ----------------------------------------------------------------------------
k    = 1.0
linf = 0.5
lam  = 1.0
Smax = 5.0
T    = 1.0

def W(q):    return 0.5 * k * q * q
def ell(S):  return linf * (1.0 - math.exp(-lam * S))
def E(q, S): return W(q) - q * ell(S)
def Sfun(t): return 0.5 * Smax * (1.0 - math.cos(2.0 * math.pi * t / T))

ELL_MAX = ell(Smax)                     # peak loading, attained at t = T/2

# ----------------------------------------------------------------------------
#  Exact solution (closed form)  and  return-map integrator
# ----------------------------------------------------------------------------
def q_exact(t, rho):
    r = rho / k
    u = ell(Sfun(t)) / k
    u_max = ELL_MAX / k
    q_peak = max(0.0, u_max - r)
    if t <= 0.5 * T:
        return max(0.0, u - r)          # ascending branch
    return min(q_peak, u + r)           # descending branch

def return_map(q_k, S_next, rho):
    f_tr = ell(S_next) - k * q_k
    if abs(f_tr) <= rho:
        return q_k                              # elastic lock
    if f_tr > rho:
        return (ell(S_next) - rho) / k          # forward flow  (f = +rho)
    return (ell(S_next) + rho) / k              # reverse flow  (f = -rho)

def run(N, rho, q0=0.0):
    h = T / N
    ts = [i * h for i in range(N + 1)]
    Ss = [Sfun(t) for t in ts]
    qs = [q0]
    diss = work = 0.0
    for i in range(N):
        q_next = return_map(qs[i], Ss[i + 1], rho)
        qs.append(q_next)
        diss += rho * abs(q_next - qs[i])
        work += E(qs[i], Ss[i + 1]) - E(qs[i], Ss[i])   # frozen-state discrete work
    defect = (E(qs[0], Ss[0]) + work) - (E(qs[N], Ss[N]) + diss)
    return ts, Ss, qs, defect

# ----------------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
#  Regimes
# ----------------------------------------------------------------------------
RHO_REV = 0.10     # reversible:   l_max = %.3f > 2 rho = 0.20
RHO_IRR = 0.30     # irreversible: l_max/2 < rho < l_max

for rho, tag in [(RHO_REV, "rev"), (RHO_IRR, "irr")]:
    ts, Ss, qs, _ = run(2000, rho)
    rows_t = [(t, S, q, ell(S)) for t, S, q in zip(ts, Ss, qs)]
    rows_l = [(ell(S), q) for S, q in zip(Ss, qs)]
    write_table(f"fig_time_{tag}.dat", "t  S  q  ell", subsample(rows_t))
    write_table(f"fig_loop_{tag}.dat", "ell  q", subsample(rows_l))

# ----------------------------------------------------------------------------
#  Convergence study (reversible regime; peak T/2 is a node for even N)
# ----------------------------------------------------------------------------
rows = []
for N in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
    ts_h, Ss_h, qs_h, defect = run(N, RHO_REV)
    err = max(abs(qs_h[i] - q_exact(ts_h[i], RHO_REV)) for i in range(len(ts_h)))
    rows.append((T / N, err, abs(defect)))
write_table("fig_convergence.dat", "h  err_state  energy_defect", rows)

# ----------------------------------------------------------------------------
#  Console summary
# ----------------------------------------------------------------------------
def peaks(rho):
    r = rho / k
    q_peak = max(0.0, ELL_MAX / k - r)
    q_res = r if ELL_MAX > 2 * rho else q_peak
    return q_peak, q_res

print(f"l_max = {ELL_MAX:.6f}")
for rho, name in [(RHO_REV, "reversible"), (RHO_IRR, "irreversible")]:
    qp, qr = peaks(rho)
    _, _, qs, _ = run(4000, rho)
    print(f"[{name:12s}] rho={rho}: q_peak(exact)={qp:.5f} num={max(qs):.5f} | "
          f"q_res(exact)={qr:.5f} num={qs[-1]:.5f}")
print("convergence (h, state error, energy defect):")
for h, e, d in rows:
    print(f"   h={h:.3e}  err_state={e:.2e}  E_defect={d:.3e}")
