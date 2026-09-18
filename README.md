# quant-research-month-1

Quant research workspace for month 1.

## Ken French data

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
