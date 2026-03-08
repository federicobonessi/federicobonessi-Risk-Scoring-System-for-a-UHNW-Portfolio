# UHNW Portfolio Risk Scoring Model

**A multi-dimensional risk assessment framework for Ultra High Net Worth portfolios.**

Built as part of [The Meridian Playbook](https://themeridianplaybook.com) — a research project on capital allocation, portfolio strategy and global financial systems.

---

## What It Does

This model scores a UHNW portfolio across five risk dimensions, producing a composite risk score (0–100) and a full visual report.

### Risk Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Market Risk | 30% | Portfolio volatility, beta exposure, risky asset concentration |
| Liquidity Risk | 25% | Weighted average days-to-liquidate, illiquid asset concentration |
| Concentration Risk | 20% | Herfindahl-Hirschman Index (HHI) at asset and class level |
| Currency Risk | 10% | Foreign currency exposure, number of distinct currencies |
| Tail Risk | 15% | Parametric VaR and CVaR at 99% confidence |

### Composite Score Scale

| Score | Rating |
|-------|--------|
| 0–24 | 🟢 LOW |
| 25–44 | 🟡 MODERATE |
| 45–64 | 🟠 ELEVATED |
| 65–79 | 🔴 HIGH |
| 80–100 | 🚨 CRITICAL |

---

## Output

The model generates a single high-resolution report (`outputs/risk_scoring_report.png`) with:

1. Composite risk gauge
2. Risk scores by dimension
3. Portfolio allocation breakdown
4. Liquidity profile
5. Monte Carlo stress test (10,000 simulations, 1-year horizon)
6. AUM distribution with VaR / CVaR
7. Full risk dimension summary table

---

## Default Portfolio (UHNW — $25M AUM)

| Asset | Weight | Liquidity |
|-------|--------|-----------|
| Global Equities | 22% | Liquid |
| Private Equity | 18% | Illiquid (5Y) |
| Real Estate | 15% | Illiquid (1Y) |
| Hedge Funds | 12% | Semi-liquid (90d) |
| Fixed Income | 13% | Liquid |
| Gold | 7% | Liquid |
| Emerging Markets | 8% | Liquid |
| Cash & Equivalents | 5% | Liquid |

---

## Installation

```bash
git clone https://github.com/your-username/risk-scoring-model.git
cd risk-scoring-model
pip install -r requirements.txt
python src/risk_scoring_model.py
```

---

## Customisation

Edit the `PORTFOLIO` DataFrame and `TOTAL_AUM` in the config section to model your own portfolio. You can adjust asset weights, volatilities, liquidity profiles, and currency exposures.

```python
PORTFOLIO = pd.DataFrame({
    "Asset":          [...],
    "Class":          [...],
    "Currency":       [...],
    "Weight":         [...],   # must sum to 1.0
    "Ann_Return":     [...],
    "Ann_Vol":        [...],
    "Liquidity_Days": [...],   # estimated days to liquidate
    "Beta":           [...],
})
```

---

## Methodology Notes

**HHI (Herfindahl-Hirschman Index)** measures portfolio concentration. A perfectly equal-weight portfolio of n assets has HHI = 1/n. Values above 0.20 indicate meaningful concentration.

**Parametric VaR/CVaR** assumes normally distributed returns. For non-normal distributions (private equity, hedge funds), actual tail risk may be higher.

**Monte Carlo Stress Test** uses Geometric Brownian Motion with portfolio-level drift and volatility. Correlations between assets are not modelled at this stage — an extension planned for v2.

---

## Context

This project is part of my broader research on **capital allocation and risk management** at [The Meridian Playbook](https://themeridianplaybook.com).

The framework reflects how a private banker or family office CIO thinks about portfolio risk — not just volatility, but liquidity, concentration, currency exposure, and tail scenarios simultaneously.

---

*Federico Bonessi — MSc Finance, IÉSEG School of Management*
*[LinkedIn](https://www.linkedin.com/in/federico-bonessi/) | [The Meridian Playbook](https://themeridianplaybook.com)*
