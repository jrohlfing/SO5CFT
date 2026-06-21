"""Regenerate the two SO(5)_1 manuscript figures from committed result data,
with publication-quality labels (no internal T1/T3/T4 names)."""
import json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(f"{REPO}/results/so5/results_so5.json"))

def chord(L, N):
    return (N/np.pi)*np.sin(np.pi*np.asarray(L)/N)

# ----- Figure 1: (left) central-charge scaling, (right) edge-mode parity -----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

colors = {"n1": "#b03030", "n2": "#3a7d44", "n4": "#d08010", "n5": "#1f3a93"}
labels = {"n1": r"$n=1$ ($c_{\rm eff}=0.50$)", "n2": r"$n=2$ ($c_{\rm eff}=1.01$)",
          "n4": r"$n=4$ ($c_{\rm eff}=2.01$)", "n5": r"$n=5$ ($c_{\rm eff}=2.51$)"}
for key in ["n1", "n2", "n4", "n5"]:
    d = J["step2"][key]
    L = np.array(d["L_values"]); S = np.array(d["S"])
    # chain length N: infer from L range (open-chain CC uses N = max usable); use stored if present
    N = d.get("L_chain", L.max())
    x = np.log(chord(L, 2*L.max()))  # chord coordinate; scale cancels in slope
    axL.plot(x, S, "o-", ms=3.5, lw=1, color=colors[key], label=labels[key])
axL.set_xlabel(r"$\ln\,[(N/\pi)\sin(\pi \ell/N)]$  (Calabrese-Cardy chord)")
axL.set_ylabel(r"entanglement entropy $S(\ell)$")
axL.set_title("Free-fermion central-charge scaling")
axL.legend(loc="upper left", fontsize=8, frameon=False)
axL.grid(True, alpha=0.25)

n5 = J["step3"]["n5"]; n4 = J["step3"]["n4"]
N5 = np.array(n5["N"]); E5 = np.clip(np.abs(n5["min_abs_E"]), 1e-18, None)
N4 = np.array(n4["N"]); E4 = np.clip(np.abs(n4["min_abs_E"]), 1e-18, None)
axR.semilogy(N5, E5, "o-", color="#1f3a93", label=r"$n=5$ (odd): protected mode")
axR.semilogy(N4, E4, "s--", color="#d08010", label=r"$n=4$ (even): no protected mode")
axR.set_xlabel(r"chain length $N$")
axR.set_ylabel(r"edge-mode splitting $\min|E|$")
axR.set_title("Unpaired-Majorana parity signature")
axR.legend(loc="center right", fontsize=8, frameon=False)
axR.grid(True, alpha=0.25, which="both")
plt.tight_layout()
plt.savefig("figures/fig1.png", dpi=200)
plt.close()
print("fig1 written")

# ----- Figure 2: (left) phase heatmap n=5, (right) Gross-Neveu scan -----
def load_grid(path):
    rows = list(csv.reader(open(path)))
    w = np.array([float(x) for x in rows[0][1:]])
    mu = np.array([float(r[0]) for r in rows[1:]])
    Z = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return mu, w, Z

mu, w, Z5 = load_grid(f"{REPO}/results/so5/phase/minE_grid_n5.csv")
fig, (axH, axG) = plt.subplots(1, 2, figsize=(11, 4.4))
logZ = np.log10(np.clip(np.abs(Z5), 1e-18, None))
im = axH.pcolormesh(w, mu, logZ, shading="auto", cmap="viridis")
axH.axhline(2.0, color="white", lw=1, ls="--", alpha=0.8)
axH.set_xlabel(r"inter-flavor coupling $w/t$")
axH.set_ylabel(r"chemical potential $\mu/t$")
axH.set_title(r"$n=5$ edge mode across $(\mu,w)$")
cb = fig.colorbar(im, ax=axH); cb.set_label(r"$\log_{10}\min|E|$")
axH.text(0.5, 2.08, "bulk gap closes", color="white", fontsize=7, ha="center")

scan = J["step4"]["scan"]
g = np.array([s["g"] for s in scan])
ce = np.array([s["c_eff"] for s in scan])
sm = np.array([s["S_mid"] for s in scan])
axG.plot(g, ce, "o-", color="#1f3a93", label=r"$c_{\rm eff}(g)$")
axG.axhline(2.5, color="#b03030", ls=":", lw=1, label=r"$SO(5)_1$ target $c=5/2$")
axG.set_xlabel(r"Gross-Neveu coupling $g$")
axG.set_ylabel(r"$c_{\rm eff}$")
axG.set_title("Marginality of the $SO(5)$ coupling")
axG.legend(loc="upper right", fontsize=8, frameon=False)
axG.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("figures/fig2.png", dpi=200)
plt.close()
print("fig2 written")

# ----- Figure 3: matched entanglement-spectrum tower (from symmetric-dmrg) -----
MT = json.load(open(f"{REPO}/symmetric-dmrg/results/matched_tower.json"))
A = MT["deliverable_A"]
fig, (ax5, ax4) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for key, title, ax, shell_col in [
        ("n5", r"$SO(5)_1$  ($n=5$, odd)", ax5, "#1f3a93"),
        ("n4", r"$Spin(4)_1$  ($n=4$, even)", ax4, "#d08010")]:
    d = A[key]
    degs = d["shell_degeneracies"][:2]          # [lowest, next], read not hardcoded
    zw = d["zero_weight_present"]
    for level, deg in enumerate(degs):          # level 0 = lowest, 1 = next
        xs = np.arange(deg) - (deg - 1) / 2.0
        for j, x in enumerate(xs):
            fc = "none"
            if level == 0 and zw and j == deg // 2:
                fc = "#b03030"                  # filled = zero-weight singlet
            ax.scatter(x, level, s=95, marker="o", edgecolors=shell_col,
                       facecolors=fc, linewidths=1.5, zorder=3)
        ax.text(0, level + 0.18, f"{deg}-fold", ha="center", fontsize=8,
                color=shell_col)
    ann = "zero-weight singlet present" if zw else "no zero-weight state"
    ax.annotate(ann, xy=(0, 0), xytext=(0, -0.5), ha="center", fontsize=8.5,
                color=("#b03030" if zw else "#555"))
    ax.set_title(title)
    ax.set_xlabel("SO weight multiplet")
    ax.set_ylim(-0.8, 1.6)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["lowest", "next"])
    ax.grid(True, axis="y", alpha=0.25)
ax5.set_ylabel("entanglement level / shell")
fig.suptitle("Matched entanglement-spectrum tower (Reshetikhin point, N=96)")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("figures/fig3.png", dpi=200)
plt.close()
print("fig3 written")
