"""
UHNW Portfolio Risk Scoring Model
===================================
A multi-dimensional risk assessment framework for Ultra High Net Worth
portfolios, covering market risk, liquidity risk, concentration risk,
currency risk, and tail risk.

Author: Federico Bonessi | The Meridian Playbook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
DARK_BG = "#0d1117"
GOLD    = "#c9a84c"
WHITE   = "#e6edf3"
GREY    = "#30363d"
RED     = "#f85149"
GREEN   = "#3fb950"
BLUE    = "#58a6ff"
ORANGE  = "#ffa657"
PURPLE  = "#a371f7"

# ─────────────────────────────────────────────
# SAMPLE UHNW PORTFOLIO
# ─────────────────────────────────────────────

PORTFOLIO = pd.DataFrame({
    "Asset":        ["Global Equities", "Private Equity", "Real Estate",
                     "Hedge Funds", "Fixed Income", "Gold",
                     "Emerging Markets", "Cash & Equivalents"],
    "Class":        ["Equity", "Alternatives", "Alternatives",
                     "Alternatives", "Fixed Income", "Commodities",
                     "Equity", "Cash"],
    "Currency":     ["USD", "USD", "EUR", "USD", "EUR", "USD", "USD", "CHF"],
    "Weight":       [0.22, 0.18, 0.15, 0.12, 0.13, 0.07, 0.08, 0.05],
    "Ann_Return":   [0.12, 0.18, 0.08, 0.09, 0.04, 0.07, 0.10, 0.03],
    "Ann_Vol":      [0.18, 0.25, 0.14, 0.10, 0.08, 0.16, 0.22, 0.01],
    "Liquidity_Days": [1, 1825, 365, 90, 2, 1, 3, 1],   # days to liquidate
    "Beta":         [1.00, 1.20, 0.60, 0.40, -0.10, 0.10, 1.15, 0.00],
})

TOTAL_AUM = 25_000_000   # USD — typical UHNW starting point
BASE_CURRENCY = "USD"

# ─────────────────────────────────────────────
# RISK DIMENSIONS
# ─────────────────────────────────────────────

def score_market_risk(df: pd.DataFrame) -> dict:
    """
    Market risk score based on weighted portfolio volatility,
    beta exposure, and equity/alternatives concentration.
    """
    w   = df["Weight"].values
    vol = df["Ann_Vol"].values
    beta = df["Beta"].values

    port_vol       = np.sqrt(w @ (np.diag(vol) @ np.diag(vol)) @ w)
    weighted_beta  = w @ beta
    risky_exposure = df.loc[df["Class"].isin(["Equity", "Alternatives"]), "Weight"].sum()

    # Normalise to 0–100
    vol_score  = min(port_vol / 0.25 * 40, 40)
    beta_score = min(abs(weighted_beta) / 1.5 * 30, 30)
    exp_score  = min(risky_exposure / 0.80 * 30, 30)

    total = vol_score + beta_score + exp_score

    return {
        "score":           round(total, 1),
        "port_volatility": port_vol,
        "weighted_beta":   weighted_beta,
        "risky_exposure":  risky_exposure,
        "components":      {"Volatility": vol_score, "Beta": beta_score, "Risky Exposure": exp_score},
    }


def score_liquidity_risk(df: pd.DataFrame) -> dict:
    """
    Liquidity risk score based on weighted average days-to-liquidate
    and illiquid asset concentration.
    """
    w    = df["Weight"].values
    days = df["Liquidity_Days"].values

    wad          = w @ days                           # weighted avg days
    illiquid_pct = df.loc[df["Liquidity_Days"] > 90, "Weight"].sum()

    wad_score   = min(wad / 500 * 50, 50)
    illiq_score = min(illiquid_pct / 0.60 * 50, 50)
    total       = wad_score + illiq_score

    return {
        "score":             round(total, 1),
        "avg_days":          round(wad, 1),
        "illiquid_pct":      illiquid_pct,
        "components":        {"Avg Days-to-Liquidate": wad_score, "Illiquid Concentration": illiq_score},
    }


def score_concentration_risk(df: pd.DataFrame) -> dict:
    """
    Concentration risk via Herfindahl-Hirschman Index (HHI)
    and single-asset / single-class cap breach.
    """
    w = df["Weight"].values
    hhi = np.sum(w ** 2)   # 1/n = perfectly diversified; 1 = fully concentrated

    # class-level HHI
    class_w  = df.groupby("Class")["Weight"].sum()
    class_hhi = np.sum(class_w.values ** 2)

    max_single = w.max()

    hhi_score   = min(hhi / 0.30 * 40, 40)
    class_score = min(class_hhi / 0.40 * 30, 30)
    single_score = min(max_single / 0.35 * 30, 30)
    total = hhi_score + class_score + single_score

    return {
        "score":       round(total, 1),
        "hhi":         round(hhi, 4),
        "class_hhi":   round(class_hhi, 4),
        "max_single":  max_single,
        "components":  {"Asset HHI": hhi_score, "Class HHI": class_score, "Single Asset Cap": single_score},
    }


def score_currency_risk(df: pd.DataFrame, base: str = BASE_CURRENCY) -> dict:
    """
    Currency risk based on foreign currency exposure
    and number of distinct currencies.
    """
    fx_exposure = df.loc[df["Currency"] != base, "Weight"].sum()
    n_currencies = df["Currency"].nunique()

    fx_score  = min(fx_exposure / 0.50 * 60, 60)
    div_score = max(0, 40 - (n_currencies - 1) * 8)   # more currencies = better diversified
    total = fx_score + div_score

    return {
        "score":        round(total, 1),
        "fx_exposure":  fx_exposure,
        "n_currencies": n_currencies,
        "components":   {"FX Exposure": fx_score, "Currency Diversification": div_score},
    }


def score_tail_risk(df: pd.DataFrame, confidence: float = 0.99) -> dict:
    """
    Tail risk via parametric VaR and CVaR (Expected Shortfall)
    at 99% confidence, assuming normal returns.
    """
    w      = df["Weight"].values
    mu_p   = w @ df["Ann_Return"].values
    vol_p  = np.sqrt(w @ np.diag(df["Ann_Vol"].values ** 2) @ w)

    z      = norm.ppf(1 - confidence)
    var_1d = -(mu_p / 252 + z * vol_p / np.sqrt(252))
    cvar_1d = -(mu_p / 252 - vol_p / np.sqrt(252) * norm.pdf(z) / (1 - confidence))

    var_score  = min(var_1d / 0.04 * 50, 50)
    cvar_score = min(cvar_1d / 0.06 * 50, 50)
    total = var_score + cvar_score

    return {
        "score":   round(total, 1),
        "var_1d":  round(var_1d, 4),
        "cvar_1d": round(cvar_1d, 4),
        "confidence": confidence,
        "components": {f"VaR ({confidence:.0%})": var_score, f"CVaR ({confidence:.0%})": cvar_score},
    }


# ─────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────

WEIGHTS = {
    "Market Risk":        0.30,
    "Liquidity Risk":     0.25,
    "Concentration Risk": 0.20,
    "Currency Risk":      0.10,
    "Tail Risk":          0.15,
}

def composite_score(scores: dict) -> float:
    total = sum(scores[k]["score"] * WEIGHTS[k] for k in WEIGHTS)
    return round(total, 1)

def risk_label(score: float) -> tuple:
    if score < 25:  return "LOW",         GREEN
    if score < 45:  return "MODERATE",    GOLD
    if score < 65:  return "ELEVATED",    ORANGE
    if score < 80:  return "HIGH",        RED
    return              "CRITICAL",       "#ff0000"


# ─────────────────────────────────────────────
# MONTE CARLO STRESS TEST
# ─────────────────────────────────────────────

def stress_test(df: pd.DataFrame, n_sim: int = 10_000, horizon_days: int = 252) -> dict:
    """Simulate AUM distribution over 1-year horizon."""
    w    = df["Weight"].values
    mu_p = w @ df["Ann_Return"].values
    vol_p = np.sqrt(w @ np.diag(df["Ann_Vol"].values ** 2) @ w)

    daily_mu  = mu_p / horizon_days
    daily_vol = vol_p / np.sqrt(horizon_days)

    # GBM simulation
    returns = np.random.normal(daily_mu, daily_vol, (n_sim, horizon_days))
    paths   = TOTAL_AUM * np.cumprod(1 + returns, axis=1)

    final   = paths[:, -1]
    var_95  = np.percentile(final, 5)
    cvar_95 = final[final <= var_95].mean()

    return {
        "paths":        paths,
        "final":        final,
        "mean":         final.mean(),
        "median":       np.median(final),
        "var_95":       var_95,
        "cvar_95":      cvar_95,
        "prob_loss":    (final < TOTAL_AUM).mean(),
        "best_case":    np.percentile(final, 95),
        "worst_case":   np.percentile(final, 5),
    }


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def make_report(df, scores, comp, stress):
    fig = plt.figure(figsize=(22, 26), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.35)

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.spines[:].set_color(GREY)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(WHITE)
        if title:
            ax.set_title(title, color=GOLD, fontsize=11, fontweight="bold", pad=10)

    usd_fmt = FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
    pct_fmt = FuncFormatter(lambda x, _: f"{x:.0%}")

    label, lcolor = risk_label(comp)

    # ── TITLE
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(DARK_BG); ax0.axis("off")
    ax0.text(0.5, 0.80, "UHNW PORTFOLIO RISK SCORING MODEL",
             ha="center", color=GOLD, fontsize=22, fontweight="bold", transform=ax0.transAxes)
    ax0.text(0.5, 0.52, f"Multi-Dimensional Risk Assessment  |  AUM: ${TOTAL_AUM/1e6:.0f}M  |  Base Currency: {BASE_CURRENCY}",
             ha="center", color=WHITE, fontsize=11, transform=ax0.transAxes)
    ax0.text(0.5, 0.22, f"COMPOSITE RISK SCORE: {comp:.1f} / 100  —  {label}",
             ha="center", color=lcolor, fontsize=16, fontweight="bold", transform=ax0.transAxes)
    ax0.axhline(0.08, color=GOLD, linewidth=0.8, xmin=0.1, xmax=0.9)

    # ── RISK GAUGE (horizontal bar)
    ax_gauge = fig.add_subplot(gs[1, :])
    ax_gauge.set_facecolor(DARK_BG); ax_gauge.axis("off")
    zones = [(25, GREEN, "LOW"), (20, GOLD, "MODERATE"), (20, ORANGE, "ELEVATED"),
             (15, RED, "HIGH"), (20, "#ff0000", "CRITICAL")]
    left = 0
    for width, color, zlabel in zones:
        ax_gauge.barh(0, width, left=left, height=0.6, color=color, alpha=0.85)
        ax_gauge.text(left + width/2, -0.55, zlabel, ha="center", color=WHITE, fontsize=8)
        left += width
    ax_gauge.barh(0, 1.5, left=comp - 0.75, height=0.8, color=WHITE, zorder=5)
    ax_gauge.text(comp, 0.65, f"{comp:.1f}", ha="center", color=WHITE, fontsize=12, fontweight="bold")
    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_title("Composite Risk Score", color=GOLD, fontsize=11, fontweight="bold", pad=6)

    # ── DIMENSION SCORES — radar-style bar
    ax1 = fig.add_subplot(gs[2, 0])
    dims   = list(WEIGHTS.keys())
    vals   = [scores[d]["score"] for d in dims]
    bcolors = [risk_label(v)[1] for v in vals]
    bars = ax1.barh(dims, vals, color=bcolors, alpha=0.9)
    ax1.set_xlim(0, 100)
    for bar, val in zip(bars, vals):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", color=WHITE, fontsize=9)
    ax1.axvline(50, color=GREY, linewidth=0.8, linestyle="--")
    style_ax(ax1, "Risk Score by Dimension")
    ax1.set_xlabel("Score (0–100)", color=WHITE, fontsize=9)

    # ── PORTFOLIO ALLOCATION PIE
    ax2 = fig.add_subplot(gs[2, 1])
    colors_pie = [GOLD, BLUE, GREEN, RED, ORANGE, PURPLE, "#ffa657", WHITE]
    _, _, ats = ax2.pie(df["Weight"], labels=df["Asset"], autopct="%1.1f%%",
                        colors=colors_pie,
                        textprops={"color": WHITE, "fontsize": 7},
                        wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5})
    for at in ats: at.set_color(DARK_BG)
    style_ax(ax2, "Portfolio Allocation")

    # ── LIQUIDITY PROFILE
    ax3 = fig.add_subplot(gs[2, 2])
    liq_colors = [GREEN if d <= 5 else GOLD if d <= 90 else RED for d in df["Liquidity_Days"]]
    ax3.barh(df["Asset"], df["Weight"], color=liq_colors, alpha=0.9)
    ax3.xaxis.set_major_formatter(pct_fmt)
    ax3.set_xlabel("Portfolio Weight", color=WHITE, fontsize=9)
    patches = [mpatches.Patch(color=GREEN, label="Liquid (≤5d)"),
               mpatches.Patch(color=GOLD,  label="Semi-liquid (≤90d)"),
               mpatches.Patch(color=RED,   label="Illiquid (>90d)")]
    ax3.legend(handles=patches, fontsize=7, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    style_ax(ax3, "Liquidity Profile by Asset")

    # ── STRESS TEST — path distribution
    ax4 = fig.add_subplot(gs[3, :2])
    sample_paths = stress["paths"][:200]
    for path in sample_paths:
        ax4.plot(path, color=BLUE, alpha=0.03, linewidth=0.8)
    pctiles = {5: RED, 25: ORANGE, 50: WHITE, 75: GREEN, 95: GREEN}
    for p, c in pctiles.items():
        ax4.plot(np.percentile(stress["paths"], p, axis=0), color=c,
                 linewidth=1.5, label=f"P{p}", linestyle="--" if p != 50 else "-")
    ax4.axhline(TOTAL_AUM, color=GOLD, linewidth=1, linestyle=":", label="Initial AUM")
    ax4.yaxis.set_major_formatter(usd_fmt)
    ax4.legend(fontsize=8, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY, ncol=3)
    ax4.set_xlabel("Trading Days", color=WHITE, fontsize=9)
    ax4.set_ylabel("Portfolio Value", color=WHITE, fontsize=9)
    style_ax(ax4, f"Monte Carlo Stress Test — 10,000 Simulations | 1-Year Horizon")

    # ── FINAL AUM DISTRIBUTION
    ax5 = fig.add_subplot(gs[3, 2])
    ax5.hist(stress["final"], bins=80, color=BLUE, alpha=0.7, edgecolor=DARK_BG)
    ax5.axvline(stress["var_95"],   color=RED,    linewidth=1.5, linestyle="--", label=f"VaR 95%: ${stress['var_95']/1e6:.1f}M")
    ax5.axvline(stress["cvar_95"],  color=ORANGE, linewidth=1.5, linestyle="--", label=f"CVaR 95%: ${stress['cvar_95']/1e6:.1f}M")
    ax5.axvline(stress["mean"],     color=GREEN,  linewidth=1.5, linestyle="-",  label=f"Mean: ${stress['mean']/1e6:.1f}M")
    ax5.axvline(TOTAL_AUM,          color=GOLD,   linewidth=1.5, linestyle=":",  label=f"Initial: ${TOTAL_AUM/1e6:.0f}M")
    ax5.xaxis.set_major_formatter(usd_fmt)
    ax5.legend(fontsize=7, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    ax5.set_xlabel("Final AUM (1Y)", color=WHITE, fontsize=9)
    style_ax(ax5, "AUM Distribution at 1-Year Horizon")

    # ── COMPONENT BREAKDOWN TABLE
    ax6 = fig.add_subplot(gs[4, :])
    ax6.set_facecolor(DARK_BG); ax6.axis("off")

    headers = ["Risk Dimension", "Score", "Rating", "Key Metric", "Value", "Weight in Composite"]
    rows = []
    key_metrics = {
        "Market Risk":        ("Portfolio Volatility",  f"{scores['Market Risk']['port_volatility']:.1%}"),
        "Liquidity Risk":     ("Avg Days-to-Liquidate", f"{scores['Liquidity Risk']['avg_days']:.0f} days"),
        "Concentration Risk": ("Asset HHI",             f"{scores['Concentration Risk']['hhi']:.4f}"),
        "Currency Risk":      ("FX Exposure",           f"{scores['Currency Risk']['fx_exposure']:.1%}"),
        "Tail Risk":          ("1-Day VaR (99%)",       f"{scores['Tail Risk']['var_1d']:.2%}"),
    }
    for dim in WEIGHTS:
        s = scores[dim]["score"]
        lbl, _ = risk_label(s)
        km, kv = key_metrics[dim]
        rows.append([dim, f"{s:.1f}", lbl, km, kv, f"{WEIGHTS[dim]:.0%}"])

    col_widths = [0.22, 0.08, 0.12, 0.22, 0.14, 0.18]
    x_positions = [sum(col_widths[:i]) for i in range(len(col_widths))]

    for j, h in enumerate(headers):
        ax6.text(x_positions[j] + 0.01, 0.92, h, transform=ax6.transAxes,
                 color=GOLD, fontsize=9, fontweight="bold")

    ax6.plot([0.01, 0.99], [0.88, 0.88], color=GOLD, linewidth=0.6,
             transform=ax6.transAxes)

    for i, row in enumerate(rows):
        y = 0.78 - i * 0.14
        for j, val in enumerate(row):
            color = WHITE
            if j == 2:
                _, color = risk_label(scores[list(WEIGHTS.keys())[i]]["score"])
            ax6.text(x_positions[j] + 0.01, y, val, transform=ax6.transAxes,
                     color=color, fontsize=9)

    ax6.set_title("Risk Dimension Summary", color=GOLD, fontsize=11,
                  fontweight="bold", pad=10)

    # ── FOOTER
    fig.text(0.5, 0.005,
             "The Meridian Playbook  |  Research on Capital Allocation & Financial Systems  |  themeridianplaybook.com",
             ha="center", color=GREY, fontsize=8)

    import os; os.makedirs("outputs", exist_ok=True)
    out = "outputs/risk_scoring_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    ✓ Report saved → {out}\n")
    return out


# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(scores, comp):
    label, _ = risk_label(comp)
    print("=" * 60)
    print("  UHNW RISK SCORING — SUMMARY")
    print("=" * 60)
    for dim in WEIGHTS:
        s = scores[dim]["score"]
        lbl, _ = risk_label(s)
        print(f"  {dim:<22}  Score: {s:>5.1f}/100   [{lbl}]")
    print("-" * 60)
    print(f"  {'COMPOSITE SCORE':<22}  Score: {comp:>5.1f}/100   [{label}]")
    print("=" * 60)

    st = scores["Tail Risk"]
    print(f"\n  1-Day VaR  (99%): {st['var_1d']:.2%}  =  ${st['var_1d'] * TOTAL_AUM:,.0f}")
    print(f"  1-Day CVaR (99%): {st['cvar_1d']:.2%}  =  ${st['cvar_1d'] * TOTAL_AUM:,.0f}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   UHNW PORTFOLIO RISK SCORING MODEL          ║")
    print("║   The Meridian Playbook                      ║")
    print("╚══════════════════════════════════════════════╝\n")

    print("📐  Computing risk dimensions...")
    scores = {
        "Market Risk":        score_market_risk(PORTFOLIO),
        "Liquidity Risk":     score_liquidity_risk(PORTFOLIO),
        "Concentration Risk": score_concentration_risk(PORTFOLIO),
        "Currency Risk":      score_currency_risk(PORTFOLIO),
        "Tail Risk":          score_tail_risk(PORTFOLIO),
    }
    comp = composite_score(scores)
    print_summary(scores, comp)

    print("🎲  Running stress test...")
    stress = stress_test(PORTFOLIO)
    print(f"    Prob. of loss (1Y): {stress['prob_loss']:.1%}")
    print(f"    95th pct AUM:  ${stress['best_case']/1e6:.2f}M")
    print(f"    5th  pct AUM:  ${stress['worst_case']/1e6:.2f}M\n")

    print("📊  Generating report...")
    make_report(PORTFOLIO, scores, comp, stress)
    print("✅  Analysis complete.")
    print("    Open outputs/risk_scoring_report.png\n")


if __name__ == "__main__":
    main()
