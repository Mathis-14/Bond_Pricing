## Project Status – Bond Bootstrapping & Yield Curve

This project explores how to bootstrap a zero‑coupon yield curve from U.S. Treasuries, with an emphasis on the underlying math and code rather than on commercial market‑data feeds.
It is difficult to find real data on coupon paying bonds without an expensive API given by Bloomberg/Refenitiv etc. 
For the moment, I tried to create my self the coupon paying bonds to boostrap to the yield curve. It is not finished yet.

### Data

- **Source**: FRED (Federal Reserve Economic Data)
- **Tenors used**: 3M, 6M, 1Y, 2Y, 3Y, 5Y constant‑maturity Treasury yields
- **Loader**: `import_bond_data.py`  
  These rates are treated as the **reference “true” zero curve** for a given date.

### Current Work

- **`yield_curve.py`** builds and plots two versions of the FRED curve:
  - **Method 1 – Convex fit**: a single globally convex quadratic curve that approximates all FRED points.
  - **Method 2 – Spline interpolation**: a natural cubic spline that passes exactly through all FRED points (not constrained to be convex).
- The resulting figures are saved to the `plot/` directory and discussed in `Theory.ipynb`.

### Next Steps

- Use the FRED‑based curve as a benchmark.
- Implement and test the **bootstrapping algorithm** on a set of **synthetic Treasury‑like coupon bonds**, and compare the resulting bootstrapped zero rates to the FRED reference curve.