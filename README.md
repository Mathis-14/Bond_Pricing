## Project Status – Bond Bootstrapping & Yield Curve

This project explores how to bootstrap a zero‑coupon yield curve from U.S. Treasuries.
Initially it used only **synthetic bonds** (because real coupon‑bond data is
usually available only via paid APIs like Bloomberg or Refinitiv). It now also
uses **real market data from Refinitiv**, with FRED yields kept as an external
benchmark curve.

### Data

- **Refinitiv (market bonds, via `refinitiv-data`)**
  - **Importer**: `import_refinitiv_data.py`
  - **Universes** (each saved as one CSV under `data/refinitiv/`):
    - `US_Treasury` – US nominal Treasury notes and bonds
    - `US_Zero` – US STRIPS / zero‑coupon Treasuries
    - `DE_Bund` – German Bund/Bobl/Schatz
    - `FR_OAT` – French OATs
    - `FR_Zero` – French zero‑coupon STRIPS
    - `IT_BTP` – Italian BTPs
  - The Refinitiv‑specific usage notes and API tips live in `REFINITIV_README.md`
    (ignored by git, for local documentation only).
    
 **FRED (benchmark zero curve)**
  - **Source**: FRED (Federal Reserve Economic Data)
  - **Tenors used**: 3M, 6M, 1Y, 2Y, 3Y, 5Y constant‑maturity Treasury yields
  - **Loader**: `import_bond_data.py`  
    These rates are treated as the **reference “true” zero curve** for a given date.

### Current Work (Refinitiv‑based, main path)

- **Refinitiv integration**:
  - `import_refinitiv_data.py` downloads real sovereign bonds (US + EU) and saves them
    as raw CSVs for later preprocessing and bootstrapping.

### Legacy Components (FRED‑only / synthetic)

These scripts are kept under `legacy/` for reference and teaching, but are no longer
the main workflow now that Refinitiv data is available.

- `legacy/yield_curve.py`: zero‑curve modelling from FRED, with:
  - Method 1 – convex quadratic fit through FRED points
  - Method 2 – natural cubic spline interpolation
  - Plots saved under `plot/` and discussed in `Theory.ipynb`.
- `legacy/bond_pricing.py`: standalone bond‑pricing utilities and examples.
- `legacy/design_synth_bonds.py`: design of synthetic Treasury‑like coupon bonds.
- `legacy/plot_synthetic_bonds.py`: visualisation of those synthetic bonds.

### Next Steps

- Implement and test the **bootstrapping algorithm** on **US Treasury coupon bonds**
  imported from Refinitiv, and compare the resulting bootstrapped zero rates to:
  - the zero‑coupon curve implied by US STRIPS (`US_Zero` universe).
  - [ ] the FRED reference curve
  - [ ] Extend to DE Bunds, FR OATs
  - [ ] Nelson‑Siegel / cubic spline smoothing
  - [ ] Compare to official yield curves

---

### Pipeline

```
import_refinitiv_data.py  →  data/refinitiv/<YYYYMMDD_HHMMSS>/  →  run_zero_curves.py
```

| Script | Role |
|---|---|
| `import_refinitiv_data.py` | Fetches bonds, buckets by maturity, saves one folder per run |
| `run_zero_curves.py` | Bootstraps zero curve + STRIPS curve, compares both |

### Usage

```bash
# Import (default: 1 bond/bucket)
python import_refinitiv_data.py
python import_refinitiv_data.py --bonds-per-bucket 3

# Curve calculation (auto-picks latest import)
python run_zero_curves.py
python run_zero_curves.py --import 20260219_195115
```

### Results

→ See [`docs/first_bootstrapping_results.md`](docs/first_bootstrapping_results.md) — methodology, plots and rate tables.