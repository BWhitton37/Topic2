"""
ECO 5012B — Coursework Project
Topic 2: Trade Tensions and Financial Markets
Reference: Ferrari Minesso, Pagliari & Kurcz (2022), "Do words hurt more
than actions? The impact of trade tensions on financial markets",
Journal of Applied Econometrics, 37(6), 1138-1159.

Sections 1 & 2: data preparation, EDA, model, synthetic data, regression.

DATA SOURCE: I collect three series from FRED (Federal Reserve Economic
Data) - S&P 500, a trade-specific policy uncertainty index, and a general
policy uncertainty index. I originally planned to pull these directly with
pandas_datareader, but pip-installing packages was being blocked by
Windows on my laptop, so instead I downloaded each series as a CSV
directly from FRED's own CSV export URLs (e.g.
https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500) and read those
in with pandas. This is disclosed here rather than hidden - the collection
step was manual, but the cleaning/merging/analysis is still fully done in
Python.

Citations:
  S&P Dow Jones Indices LLC, S&P 500 [SP500], FRED, Federal Reserve Bank
    of St. Louis; https://fred.stlouisfed.org/series/SP500
  Baker, S. R., Bloom, N., & Davis, S. J., Economic Policy Uncertainty
    Index: Categorical Index: Trade Policy [EPUTRADE], FRED;
    https://fred.stlouisfed.org/series/EPUTRADE
  Baker, S. R., Bloom, N., & Davis, S. J., Economic Policy Uncertainty
    Index for United States [USEPUINDXD], FRED;
    https://fred.stlouisfed.org/series/USEPUINDXD

As a robustness check (Section 2.3b), I also re-run the same regression
on the paper's own replication data, which I was given directly, to see
whether my self-collected version tells a consistent story.

SETUP: needs pandas, numpy, scipy, matplotlib, seaborn, statsmodels, plotly.
Install with: pip install -r ../requirements.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px
from scipy import stats

RAW_DIR = "../data/raw"                  # SP500.csv, EPUTRADE.csv, USEPUINDXD.csv
REPLICATION_RAW = "../data/raw/DataDaily.csv"   # authors' replication file (robustness check)
PROCESSED_PATH = "../data/processed/trade_tension_data.csv"
SYNTHETIC_PATH = "../data/processed/synthetic_data.csv"

# =========================================================================
# SECTION 1 — DATA COLLECTION, CLEANING & EDA
# =========================================================================

# --- Step 1.1: Data acquisition -----------------------------------------
sp500 = pd.read_csv(f"{RAW_DIR}/SP500.csv", parse_dates=["observation_date"])
epu = pd.read_csv(f"{RAW_DIR}/USEPUINDXD.csv", parse_dates=["observation_date"])
tpu_monthly = pd.read_csv(f"{RAW_DIR}/EPUTRADE.csv", parse_dates=["observation_date"])

sp500 = sp500.set_index("observation_date").rename(columns={"SP500": "SP500_level"})
epu = epu.set_index("observation_date").rename(columns={"USEPUINDXD": "EPU"})
tpu_monthly = tpu_monthly.set_index("observation_date").rename(columns={"EPUTRADE": "TPU_monthly"})

# FRED marks non-trading days with blanks -> these come in as NaN. I drop
# them since they're genuinely missing observations, not zeros.
sp500 = sp500.dropna()
epu = epu.dropna()

# Decision: sample window. FRED's SP500 series only keeps a rolling ~10
# years of daily history (per their data-licensing terms with S&P Dow
# Jones), so it starts 2016-08-08, not Jan 2016. I keep the window at
# 2016-08-08 to 2019-12-31: this still fully covers both trade-war
# escalation episodes (Mar-Jul 2018 and May-Aug 2019), and I'm only
# missing the first ~7 months of 2016, which were a quiet period for
# trade tension anyway.
START, END = "2016-08-08", "2019-12-31"

# --- Step 1.2: Data preparation and cleaning -----------------------------
# Frequency alignment: SP500 and the general EPU index are daily, but the
# trade-specific index (EPUTRADE) is only published MONTHLY. This is the
# genuine forward-fill-vs-aggregate decision the brief asks about. I
# forward-fill the monthly TPU value across every trading day in that
# month, rather than aggregating the daily series down to monthly, because
# I want to keep full daily variation in returns - my model needs daily
# volatility, and aggregating to monthly would throw most of that away.
clean = sp500.join(epu, how="inner")
clean["TPU"] = tpu_monthly["TPU_monthly"].reindex(clean.index, method="ffill")
clean = clean.loc[START:END].dropna().sort_index()

# Returns and volatility proxy: SP500 here is a price LEVEL, so I take the
# log difference to get a daily return, then square it as a simple
# volatility measure (per the brief's Step 1.3 suggestion).
clean["ret"] = np.log(clean["SP500_level"]).diff() * 100
clean["ret_sq"] = clean["ret"] ** 2
clean["ret_sq_lag1"] = clean["ret_sq"].shift(1)
clean = clean.dropna().reset_index().rename(columns={"observation_date": "Time"})
clean.to_csv(PROCESSED_PATH, index=False)
print(f"Saved cleaned dataset: {clean.shape[0]} obs, {clean.Time.min().date()} to "
      f"{clean.Time.max().date()} -> {PROCESSED_PATH}")

# --- Step 1.3: Exploratory data analysis ---------------------------------
corr = clean[["ret", "ret_sq", "TPU", "EPU"]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation: returns, volatility, trade tension (TPU) & EPU")
plt.tight_layout()
plt.savefig("../figures/correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(clean["Time"], clean["ret"], color="steelblue", lw=0.7)
axes[0].set_ylabel("S&P 500 daily return (%)")
axes[0].set_title("S&P 500 returns and trade-policy uncertainty (EPUTRADE), 2016-2019")
axes[1].plot(clean["Time"], clean["TPU"], color="firebrick", lw=0.8)
axes[1].set_ylabel("Trade Policy Uncertainty (EPUTRADE)")
for ax in axes:
    ax.axvspan(pd.Timestamp("2018-03-01"), pd.Timestamp("2018-07-06"),
               color="orange", alpha=0.15, label="Tariff announcements (Mar-Jul 2018)")
    ax.axvspan(pd.Timestamp("2019-05-01"), pd.Timestamp("2019-08-01"),
               color="purple", alpha=0.15, label="Escalation (May-Aug 2019)")
axes[0].legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("../figures/timeseries_annotated.png", dpi=150)
plt.close()
print("Correlation matrix:\n", corr.round(3))

# =========================================================================
# SECTION 2 — MODEL DEVELOPMENT AND REPLICATION
# =========================================================================

# --- Step 2.1: Model specification ---------------------------------------
# I estimate a version of the paper's "Model 2.1", linking trade
# uncertainty to a market-risk variable (squared returns):
#
#     ret_sq_t = b0 + gamma*TPU_t + b1*ret_sq_(t-1) + b2*EPU_t + eps_t
#
# where:
#   ret_sq_t     = squared daily return of the S&P 500 (my volatility proxy)
#   TPU_t        = trade-specific policy uncertainty (EPUTRADE, forward-
#                  filled to daily) - my main variable of interest, gamma
#                  is the "Uncertainty Impact Coefficient" the brief asks for
#   ret_sq_(t-1) = yesterday's squared return, controlling for volatility
#                  clustering (volatile days tend to follow volatile days)
#   EPU_t        = general (non-trade) policy uncertainty, controlling for
#                  the possibility that TPU is just picking up broad policy
#                  noise rather than trade-specific tension
#   eps_t        = error term, assumed mean zero given the regressors

# --- Step 2.2: Synthetic data generation ---------------------------------
# ~200 synthetic daily observations, following the AR(1) idea from
# Lecture 9: today's volatility depends partly on yesterday's, plus a
# shock linked to TPU. I use Student's t-distributed (not Normal) random
# shocks, since real daily stock returns have more extreme days than a
# Normal distribution predicts ("fat tails"). I set the true TPU effect to
# a known positive value so I can check my regression correctly recovers
# it before trusting it on real data.
rng = np.random.default_rng(42)
n = 200
gamma_true, rho_true, sigma_base = 0.8, 0.35, 0.5

tpu_syn = rng.choice([0, 0, 0, 0, 1, -1], size=n) * rng.exponential(0.6, size=n)
volatility = np.zeros(n)
volatility[0] = sigma_base
for t in range(1, n):
    volatility[t] = max(0.01, sigma_base * (1 - rho_true) + rho_true * volatility[t-1]
                         + gamma_true * abs(tpu_syn[t]))
shock = stats.t.rvs(df=4, size=n, random_state=rng.integers(1_000_000_000))
shock = shock / np.sqrt(4 / 2)
ret_syn = np.sqrt(volatility) * shock
ret_sq_syn = ret_syn ** 2

synthetic = pd.DataFrame({"t": np.arange(n), "TPU": tpu_syn, "ret": ret_syn, "ret_sq": ret_sq_syn})
synthetic["ret_sq_lag1"] = synthetic["ret_sq"].shift(1)
synthetic = synthetic.dropna().reset_index(drop=True)
synthetic.to_csv(SYNTHETIC_PATH, index=False)

print(f"\nSynthetic data checks:")
print(f"  Autocorrelation of the TRUE volatility series: "
      f"{pd.Series(volatility).autocorr(1):.3f} (target rho={rho_true})")
print(f"  Autocorrelation of the OBSERVED squared returns: "
      f"{synthetic['ret_sq'].autocorr(1):.3f} (expect much noisier/lower)")
print(f"  Excess kurtosis of synthetic returns: {stats.kurtosis(synthetic['ret']):.2f} "
      f"(should be above 0 - more extreme values than Normal)")

X_syn = sm.add_constant(synthetic[["TPU", "ret_sq_lag1"]])
model_syn = sm.OLS(synthetic["ret_sq"], X_syn).fit()
print(f"\n=== Synthetic-data regression (true gamma = {gamma_true}) ===")
print(model_syn.summary())

# --- Step 2.3: Empirical replication (self-collected data) ---------------
X_real = sm.add_constant(clean[["TPU", "ret_sq_lag1", "EPU"]])
y_real = clean["ret_sq"]
model_real = sm.OLS(y_real, X_real).fit()
print("\n=== Main regression (self-collected FRED data) ===")
print(model_real.summary())

gamma_hat = model_real.params["TPU"]
gamma_p = model_real.pvalues["TPU"]
print(f"\nUncertainty Impact Coefficient (gamma_hat) = {gamma_hat:.6f}, p-value = {gamma_p:.4f}")
# Interpretation: gamma_hat is positive and statistically significant
# (p < 0.01). Unlike a signed sentiment index, EPUTRADE only takes
# positive values (it's a frequency count of trade-uncertainty news
# coverage, not a tone score), so the interpretation is direct: higher
# trade-policy uncertainty is associated with higher next-period market
# volatility, which lines up with the paper's central finding.

# --- Step 2.3b: Robustness check against the paper's replication data ----
# As a robustness check, I re-run the identical specification on the
# paper's own replication dataset (DataDaily.csv, which I was given
# directly rather than collecting myself - see documentation), to see
# whether an independently-collected series and the authors' own series
# tell a broadly consistent story.
try:
    rep_raw = pd.read_csv(REPLICATION_RAW, parse_dates=["Time"], dayfirst=True).sort_values("Time")
    rep_raw["ret"] = rep_raw["SP500"].diff() * 100        # SP500 col here is a log level
    rep_raw["ret_sq"] = rep_raw["ret"] ** 2
    rep = rep_raw.rename(columns={"Index_sum": "TPU_paper", "EconPolUnc": "EPU_paper"})
    rep["ret_sq_lag1"] = rep["ret_sq"].shift(1)
    rep = rep.dropna()

    X_rep = sm.add_constant(rep[["TPU_paper", "ret_sq_lag1", "EPU_paper"]])
    model_rep = sm.OLS(rep["ret_sq"], X_rep).fit()
    print("\n=== Robustness check: same model on the paper's replication data ===")
    print(model_rep.summary())
    print(f"\nComparison:")
    print(f"  Self-collected (EPUTRADE):     gamma_hat = {gamma_hat:+.4f}  p = {gamma_p:.4f}")
    print(f"  Replication data (Index_sum):  gamma_hat = {model_rep.params['TPU_paper']:+.4f}  "
          f"p = {model_rep.pvalues['TPU_paper']:.4f}")
    print("  Note: the replication data's TPU is a SIGNED tone index (negative = tense "
          "news), so its negative coefficient means the same thing as a positive "
          "coefficient on an unsigned index like EPUTRADE - see documentation.")
except FileNotFoundError:
    print("\n[Robustness check skipped: DataDaily.csv not found in ../data/raw/]")

# --- Step 2.4: Replication plot (Plotly Express) ---------------------------
clean["fitted"] = model_real.fittedvalues
plot_df = clean.melt(id_vars="Time", value_vars=["ret_sq", "fitted"],
                      var_name="series", value_name="value")
plot_df["series"] = plot_df["series"].map({"ret_sq": "Actual squared return",
                                            "fitted": "Model fitted value"})
fig = px.line(plot_df, x="Time", y="value", color="series",
              title="Actual vs. fitted squared S&P 500 return",
              labels={"value": "Squared daily return (%^2)", "Time": "Date"})
fig.update_layout(legend_title_text="")
fig.write_html("../figures/replication_plot.html")
print("\nSaved replication plot -> ../figures/replication_plot.html")

# --- Save fitted coefficients for the Streamlit app (Section 3) ----------
import json
coefs = {
    "const": float(model_real.params["const"]),
    "gamma_TPU": float(gamma_hat),
    "beta_ret_sq_lag1": float(model_real.params["ret_sq_lag1"]),
    "beta_EPU": float(model_real.params["EPU"]),
    "gamma_p_value": float(gamma_p),
}
with open("../data/processed/model_coefficients.json", "w") as f:
    json.dump(coefs, f, indent=2)
print("\nSaved fitted coefficients -> ../data/processed/model_coefficients.json")
