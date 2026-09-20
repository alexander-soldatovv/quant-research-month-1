# quant-research-month-1

Quant research workspace for month 1.

## Main notebook

Run the full assignment notebook from the repository root:

```bash
python3 -m pip install numpy pandas scipy matplotlib nbformat nbclient ipykernel
```

Then open and run:

- `01_returns_and_statistics.ipynb`

The notebook runs the full workflow end to end:

- downloads Kenneth French data programmatically;
- validates dates, missing values, duplicates, and units;
- calculates simple, log, and excess returns;
- compares industry mean, median, volatility, skewness, and kurtosis;
- builds histograms and Q-Q plots against fitted Normal and Student-t distributions;
- computes bootstrap intervals for industry means and medians;
- solves four Jane Street probability problems analytically and checks them by Monte Carlo;
- shows dependent random variables with zero correlation.

## Ken French data

Source: Kenneth French Data Library.

Download the monthly Kenneth French datasets:

```bash
python3 scripts/download_ken_french.py
```

The script downloads the original zip archives from the Kenneth French Data Library and writes:

- `data/raw/F-F_Research_Data_Factors_CSV.zip`
- `data/raw/F-F_Research_Data_Factors.csv`
- `data/raw/10_Industry_Portfolios_CSV.zip`
- `data/raw/10_Industry_Portfolios.csv`
- `data/processed/fama_french_3_factors_monthly.csv`
- `data/processed/10_industry_portfolios_monthly_vw.csv`
- `data/processed/ken_french_metadata.json`

Processed returns are monthly percent returns. The 10 industry portfolio output uses the value-weighted monthly table.
The raw Kenneth French files are preserved under `data/raw/`; cleaned monthly tables are written under `data/processed/`.
No manual download, Excel editing, or hand cleaning is required.

Validate the cleaned files:

```bash
python3 scripts/validate_ken_french.py
```

Calculate simple, logarithmic, and excess returns:

```bash
python3 scripts/calculate_returns.py
```

This writes:

- `data/processed/fama_french_3_factors_returns.csv`
- `data/processed/10_industry_portfolios_returns.csv`

Calculated return files use decimal returns. For example, `2.89%` is written as `0.0289`.
Log returns are calculated as `log(1 + simple_return)`.
Excess simple returns are calculated as `simple_return - RF_simple`.
Excess log returns are calculated as `log_return - RF_log`.

Compare descriptive statistics across industries:

```bash
python3 scripts/compare_industry_stats.py
```

This writes:

- `data/processed/10_industry_portfolios_stats.csv`

The statistics are calculated for `simple`, `log`, and `excess_simple` monthly decimal returns.
Volatility is the sample standard deviation of monthly returns.
Kurtosis is reported as excess kurtosis, so a normal distribution is approximately `0`.

Plot histograms and Q-Q diagnostics for industry simple returns:

```bash
python3 scripts/plot_industry_distributions.py
```

This writes PNG files to:

- `reports/figures/industry_simple_return_histograms.png`
- `reports/figures/*_distribution_diagnostics.png`

Calculate bootstrap confidence intervals for industry means and medians:

```bash
python3 scripts/bootstrap_industry_intervals.py
```

This writes:

- `data/processed/10_industry_portfolios_bootstrap_intervals.csv`
- `reports/figures/industry_bootstrap_intervals_simple.png`

The script uses a percentile bootstrap with 10,000 resamples and a fixed random seed.

Run the Jane Street Monte Carlo checks directly:

```bash
python3 scripts/jane_street_monte_carlo.py
```

The four checked answers are:

- maximum of three d6 rolls: `119/24`;
- one optional d6 reroll: optimal reroll on `1,2,3`, expected value `17/4`;
- expected flips until `HHH`: `14`;
- `B+R` and `B*R` are not independent.
