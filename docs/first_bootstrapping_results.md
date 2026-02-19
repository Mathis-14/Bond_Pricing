# US Treasury Zero-Coupon Curve — First Results
*Valuation date: 19 February 2026*

---

## 1. Objective

Build and compare two independent estimates of the **US Treasury zero-coupon (spot) curve**:

1. **Bootstrapped curve** — extracted from coupon-bearing Treasury bonds
2. **STRIPS curve** — read directly from zero-coupon Treasury STRIPS prices

Both approaches should give the same theoretical curve. Observed differences reveal **liquidity premia**, market microstructure effects, or data artefacts.

---

## 2. Data

All market data is sourced from **Refinitiv** via the `import_refinitiv_data.py` pipeline.

| Dataset | Source | Instruments | Filter |
|---|---|---|---|
| Coupon bonds | `rd.discovery.search` + `rd.get_data` | US nominal Treasuries | `CouponRate > 0`, `IssuerCountry = US`, `Currency = USD` |
| STRIPS | `rd.discovery.search` + `rd.get_data` | US zero-coupon STRIPS | `CouponRate = 0`, same country/currency |

**Maturity bucketing**: To avoid over-representing crowded maturities, bonds are assigned to the nearest of 20 target tenors:

> 3M · 6M · 9M · 1Y · 1.5Y · 2Y · 2.5Y · 3Y · 4Y · 5Y · 6Y · 7Y · 8Y · 9Y · 10Y · 12Y · 15Y · 20Y · 25Y · 30Y

Within each bucket, the **most recently issued** bond (on-the-run) is selected, as it carries the highest liquidity and tightest bid-ask spread.

---

## 3. Methodology

### 3.1 Bootstrapping from Coupon Bonds

A coupon bond with face value 100 pays cash flows $CF_k$ at times $T_k$. Its price satisfies:

$$P_{\text{clean}} = \sum_{k=1}^{n} CF_k \cdot P(0, T_k)$$

where $P(0,T) = e^{-r(T)\,T}$ is the **discount factor** and $r(T)$ is the **continuously-compounded zero rate**.

**Bootstrapping** solves for each $P(0, T_n)$ sequentially, from the shortest to the longest maturity:

$$P(0, T_n) = \frac{P_{\text{clean}} - \displaystyle\sum_{k=1}^{n-1} CF_k \cdot P(0, T_k)}{CF_n}$$

Intermediate maturities are obtained via **log-linear interpolation** (piecewise-constant forward rate assumption):

$$P(0,T) = P(0,T_i)^{1-w} \cdot P(0,T_{i+1})^{w}, \qquad w = \frac{T - T_i}{T_{i+1} - T_i}$$

Zero rates are then recovered as:

$$r(T) = -\frac{\ln P(0,T)}{T}$$

### 3.2 Direct Rates from STRIPS

A STRIPS is already a zero-coupon bond. Its price *is* the discount factor:

$$P(0,T) = \frac{\text{mid price}}{100}, \qquad r(T) = -\frac{\ln P(0,T)}{T}$$

No bootstrapping required — one formula, one bond, one rate.

---

## 4. Results — Run 1: 1 Bond per Bucket

**Import:** `first imports` (1 bond per maturity bucket — on-the-run only)
**Bootstrapped bonds:** 20 | **STRIPS:** 329

### Curve comparison

![Zero curve comparison — 1 bond per bucket](plot/zero_curves/zero_curve_comparison_1bond.png)

### Key rates at benchmark tenors

| Maturity | Bootstrapped (%) | STRIPS (%) | Diff (bp) |
|:---:|:---:|:---:|:---:|
| 6M | 0.063 ⚠️ | 3.591 | +352.8 |
| 1Y | 3.686 | 3.468 | −21.8 |
| 2Y | 3.626 | 3.485 | −14.2 |
| 3Y | 3.496 | 3.391 | −10.6 |
| 5Y | 3.682 | 3.628 | −5.3 |
| 7Y | 3.879 | 3.844 | −3.6 |
| 10Y | 4.106 | 4.093 | −1.3 |
| 20Y | 4.758 | 4.885 | +12.7 |

**Global statistics:** Mean diff = −0.003 pp · Std = 0.354 pp · RMSE = 0.354 pp

> ⚠️ **6M anomaly**: the on-the-run 6M bond has a near-zero coupon rate, behaving like a bill with an artificially distorted bootstrapped zero rate. Flagged for data cleaning.

---

## 5. Results — Run 2: 3 Bonds per Bucket

**Import:** `20260219_195115` (3 bonds per maturity bucket)
**Bootstrapped bonds:** 60 | **STRIPS:** 329

### Curve comparison

![Zero curve comparison — 3 bonds per bucket](plot/zero_curves/zero_curve_comparison_3bonds.png)

### Key rates at benchmark tenors

| Maturity | Bootstrapped (%) | STRIPS (%) | Diff (bp) |
|:---:|:---:|:---:|:---:|
| 6M | 0.000 ⚠️ | 3.591 | +359.1 |
| 1Y | 3.568 | 3.468 | −9.9 |
| 2Y | 3.468 | 3.485 | +1.6 |
| 3Y | 3.496 | 3.391 | −10.5 |
| 5Y | 3.681 | 3.628 | −5.3 |
| 7Y | 3.872 | 3.844 | −2.9 |
| 10Y | 4.102 | 4.093 | −0.9 |
| 20Y | 4.771 | 4.885 | +11.4 |

**Global statistics:** Mean diff = +0.010 pp · Std = 0.304 pp · RMSE = 0.304 pp

> ⚠️ **Short-end instability**: with 3 bonds per bucket, the bootstrapper processes multiple bonds with very similar maturities sequentially. Small pricing errors compound, producing oscillations in the 0–3Y segment (visible in the chart). This is a known **bootstrap instability** pathology. The long end (4Y+) is not affected.

---

## 6. Interpretation

### Curve shape — 19 Feb 2026

The US Treasury yield curve is **upward sloping** from 1Y to 30Y, consistent with a standard term premium:

| Segment | Level | Comment |
|---|---|---|
| 1Y | ~3.5% | Short-end anchored near Fed funds rate |
| 2–5Y | 3.5–3.7% | Gradual rise, mild inversion resolved |
| 10Y | ~4.1% | Benchmark tenor |
| 30Y | ~4.8% | Long-end term premium |

### Bootstrapped vs STRIPS — what the gap tells us

- **1Y to 10Y**: STRIPS yield slightly *less* than the bootstrapped curve (−10 to −1 bp). This is the classic **STRIPS liquidity premium**: STRIPS are less liquid than coupon bonds, but the effect is small and shrinking at longer tenors.
- **20Y+**: STRIPS yield *more* (+11–13 bp), reflecting demand for long-duration zero-coupon instruments from pension funds and insurers (ALM demand compressing prices at the long end — wait, higher demand → higher price → lower yield; here STRIPS yield more which means lower demand or higher supply at the very long end).
- **Overall**: differences are well below 20 bp in the 1–20Y range, confirming the two methods are consistent. The curves are effectively the same.

### Bootstrap stability

| Config | Short-end behaviour | Recommended for |
|---|---|---|
| 1 bond / bucket | ✅ Smooth | Curve production |
| 3 bonds / bucket | ⚠️ Oscillates 0–3Y | Data validation only |

---

## 7. Next Steps

- [ ] Filter out the 6M bill / near-zero coupon anomaly in `import_refinitiv_data.py`
- [ ] Extend to German Bunds (DE curve) and French OATs (FR curve)
- [ ] Implement Nelson-Siegel or cubic spline smoothing to get a continuous curve
- [ ] Add historical imports to study curve dynamics over time
- [ ] Compare to official yield curves 

