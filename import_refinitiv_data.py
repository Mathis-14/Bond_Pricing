"""
Download and prepare sovereign bond data from Refinitiv.

This is the single script that handles ALL Refinitiv API interactions:
  1. Opens a Refinitiv session using REFINITIV_API_KEY from .env
  2. Downloads raw bond universes via discovery.search + get_data
  3. Fetches maturity dates via the bond IPA layer (bond.Definition)
     when rd.get_data does not return them
  4. Prepares canonical datasets ready for bootstrapping:
     - us_treasury_input_YYYYMMDD.csv  (coupon bonds, one per maturity bucket)
     - us_strips_input_YYYYMMDD.csv    (STRIPS with prices + maturities)
  5. Closes the session

After running this script, no downstream code needs Refinitiv access.


The Maturity Bucket Parameter

The Maturity Bucket Parameter is a way to group bonds by maturity. It is used to create a clean bootstrapping process.
The US Treasury market has many bonds per maturity zone for example, there might be 15 different bonds with 4.5 to 
5.5 years remaining. If you usethem into the bootstrapper => noisy points concentrated in 
popular maturities while other tenors might be sparse.

Visual example : 

Bonds found:        4.6Y   4.8Y   5.0Y   5.1Y   5.4Y   9.2Y   9.7Y   10.1Y
                     ↓      ↓      ↓      ↓      ↓      ↓      ↓       ↓
Assigned bucket:   [5.0]  [5.0]  [5.0]  [5.0]  [5.0]  [9.0]  [10.0] [10.0]
                         ↓                              ↓       ↓
Selected (newest):  1 bond kept                  1 bond kept  1 bond kept



# Use 3 bonds per bucket
python import_refinitiv_data.py --bonds-per-bucket 3

# Use 1 bond (on-the-run only) — same as the default
python import_refinitiv_data.py --bonds-per-bucket 1

# Use ALL bonds (equivalent to BONDS_PER_BUCKET = None)
python import_refinitiv_data.py --bonds-per-bucket 0

"""

from __future__ import annotations

import argparse
import os
import re
import datetime
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
import refinitiv.data as rd

try:
    from refinitiv.data.content.ipa.financial_contracts import bond as bond_ipa
    _IPA_AVAILABLE = True
except ImportError:
    bond_ipa = None
    _IPA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
REFINITIV_DIR = DATA_DIR / "refinitiv"
BOOTSTRAP_DIR = DATA_DIR / "bootstrapped"


def _make_import_dir(timestamp: str) -> Path:
    """Create and return data/refinitiv/<timestamp>/ for this import run."""
    import_dir = REFINITIV_DIR / timestamp
    import_dir.mkdir(parents=True, exist_ok=True)
    return import_dir

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------
BOND_UNIVERSES = {
    "US_Treasury": {
        "description": "US Treasury Bonds and Notes (excluding TIPS)",
        "filter": (
            "IsActive eq true and "
            "DbType eq 'GOVT' and "
            "RCSAssetCategoryLeaf eq 'bond' and "
            "Currency eq 'USD' and "
            "IssuerCountry eq 'US' and "
            "CouponRate gt 0"
        ),
    },
    "US_Zero": {
        "description": "US Zero Coupon Bonds / STRIPS",
        "filter": (
            "IsActive eq true and "
            "DbType eq 'GOVT' and "
            "RCSAssetCategoryLeaf eq 'bond' and "
            "Currency eq 'USD' and "
            "IssuerCountry eq 'US' and "
            "CouponRate eq 0"
        ),
    },
    "DE_Bund": {
        "description": "German Government Bonds (Bunds, Bobls, Schatz)",
        "filter": (
            "IsActive eq true and "
            "DbType eq 'GOVT' and "
            "RCSAssetCategoryLeaf eq 'bond' and "
            "Currency eq 'EUR' and "
            "IssuerCountry eq 'DE' and "
            "CouponRate gt 0"
        ),
    },
    "FR_OAT": {
        "description": "French Government Bonds (OATs)",
        "filter": (
            "IsActive eq true and "
            "DbType eq 'GOVT' and "
            "RCSAssetCategoryLeaf eq 'bond' and "
            "Currency eq 'EUR' and "
            "IssuerCountry eq 'FR' and "
            "CouponRate gt 0"
        ),
    },
    "IT_BTP": {
        "description": "Italian Government Bonds (BTPs)",
        "filter": (
            "IsActive eq true and "
            "DbType eq 'GOVT' and "
            "RCSAssetCategoryLeaf eq 'bond' and "
            "Currency eq 'EUR' and "
            "IssuerCountry eq 'IT' and "
            "CouponRate gt 0"
        ),
    },
    "FR_Zero": {
        "description": "French Zero Coupon Bonds (e.g. OATs STRIPS)",
        "filter": (
            "IsActive eq true and "
            "DbType eq 'GOVT' and "
            "RCSAssetCategoryLeaf eq 'bond' and "
            "Currency eq 'EUR' and "
            "IssuerCountry eq 'FR' and "
            "CouponRate eq 0"
        ),
    },
}

RAW_FIELDS = [
    "ISIN", "RIC", "Ticker",
    "TR.CommonName", "TR.FiCurrency", "TR.AssetStatus", "TR.IssuerCountry",
    "TR.BidPrice", "TR.AskPrice", "TR.MidPrice", "TR.Yield",
    "TR.CouponRate", "TR.InterestPaymentFrequency", "TR.DayCountBasis",
    "TR.IssueDate", "TR.MaturityDate", "TR.FirstCouponDate",
]

RENAME_RAW = {
    "Instrument": "RIC",
    "Company Common Name": "Name",
    "Currency": "Currency",
    "Status": "Status",
    "Bid Price": "BidPrice",
    "Ask Price": "AskPrice",
    "Mid Price": "MidPrice",
    "Yield": "Yield",
    "Coupon Rate": "CouponRate",
    "Issue Date": "IssueDate",
    "TICKER": "Ticker",
    "Maturity Date": "MaturityDate",
    "Interest Payment Frequency": "CouponFrequency",
    "Day Count Basis": "DayCountBasis",
    "First Coupon Date": "FirstCouponDate",
}

MATURITY_BUCKETS = [
    0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0,
]

# Number of bonds to select per maturity bucket (sorted by IssueDate, newest first)
# Set to None to use ALL bonds (no bucket filtering)
BONDS_PER_BUCKET = 1

# ============================================================================
# Helpers
# ============================================================================

def _parse_frequency_to_int(freq) -> int:
    """Map InterestPaymentFrequency to integer payments/year. Default 2 (semi-annual)."""
    if freq is None or (isinstance(freq, float) and pd.isna(freq)):
        return 2
    if isinstance(freq, (int, float)) and not pd.isna(freq):
        v = int(freq)
        if 1 <= v <= 12:
            return v
    s = str(freq).strip().upper()
    mapping = {
        "1M": 12, "1MO": 12,
        "2M": 6, "2MO": 6,
        "3M": 4, "3MO": 4, "Q": 4, "QUARTERLY": 4,
        "4M": 3, "4MO": 3,
        "6M": 2, "6MO": 2, "SA": 2, "SEMI-ANNUAL": 2, "SEMIANNUAL": 2,
        "12M": 1, "1Y": 1, "1YR": 1, "A": 1, "ANNUAL": 1,
    }
    if s in mapping:
        return mapping[s]
    m = re.match(r"^(\d+)\s*[MmoO]$", s)
    if m:
        months = int(m.group(1))
        if months > 0:
            return max(1, round(12 / months))
    return 2


def _T_years(maturity: date, valuation: date) -> float:
    return max((maturity - valuation).days, 0) / 365.25


def _bucket(T: float) -> float:
    if T <= 0:
        return MATURITY_BUCKETS[0]
    return min(MATURITY_BUCKETS, key=lambda b: abs(b - T))


def _ipa_get_data(ric: str, fields: List[str]) -> pd.DataFrame | None:
    """Call bond IPA Definition for a single RIC. Returns one-row DataFrame or None."""
    if not _IPA_AVAILABLE or bond_ipa is None:
        return None
    try:
        kwargs = {"instrument_code": ric, "fields": fields}
        try:
            kwargs["outputs"] = [
                rd.ipa.financial_contracts.Outputs.DATA,
                rd.ipa.financial_contracts.Outputs.HEADERS,
            ]
        except AttributeError:
            pass
        resp = bond_ipa.Definition(**kwargs).get_data()
    except Exception:
        return None
    if resp is None or not hasattr(resp, "data") or resp.data is None:
        return None
    df = getattr(resp.data, "df", None)
    if df is None or df.empty:
        return None
    return df


# ============================================================================
# Step 1: Download raw universes
# ============================================================================

def _download_raw_universes(timestamp: str, import_dir: Path) -> dict[str, Path]:
    """Search + get_data for each universe. Returns {name: csv_path}."""
    saved = {}

    for name, cfg in BOND_UNIVERSES.items():
        print(f"\n--- {name}: {cfg['description']} ---")
        try:
            search_df = rd.discovery.search(
                view=rd.discovery.Views.GOV_CORP_INSTRUMENTS,
                filter=cfg["filter"],
                select="RIC, DTSubjectName",
                top=2000,
            )
            if search_df is None or search_df.empty:
                print(f"  No instruments found.")
                continue

            rics = search_df["RIC"].dropna().astype(str).tolist()
            if not rics:
                continue
            print(f"  Found {len(rics)} instruments.")

            df = rd.get_data(universe=rics, fields=RAW_FIELDS)
            if df is None or df.empty:
                print(f"  No data returned.")
                continue

            path = import_dir / f"{name}_{timestamp}.csv"
            df.to_csv(path, index=False)
            saved[name] = path
            print(f"  Saved {len(df)} rows -> {path}")
        except Exception as e:
            print(f"  Error: {e}")

    return saved


# ============================================================================
# Step 2: Prepare US Treasury coupon bonds
# ============================================================================

def _prepare_us_treasuries(raw_path: Path, timestamp: str, import_dir: Path) -> Path:
    """
    Filter nominal GOVT bonds, fetch maturities via IPA, select bonds per
    maturity bucket (controlled by BONDS_PER_BUCKET). Save as us_treasury_input_<timestamp>.csv.
    """
    print("\n=== Preparing US Treasury coupon bonds ===")
    raw = pd.read_csv(raw_path).rename(columns=RENAME_RAW)

    # Filter to GOVT with valid prices
    raw = raw.dropna(subset=["RIC", "MidPrice"]).copy()
    raw["Ticker"] = raw["Ticker"].astype(str)
    nominal = raw[raw["Ticker"] == "GOVT"].copy()
    if nominal.empty:
        raise RuntimeError("No nominal GOVT Treasuries found.")
    print(f"  {len(nominal)} nominal GOVT bonds after filtering.")

    rics = nominal["RIC"].dropna().astype(str).unique().tolist()
    price_by_ric = dict(zip(nominal["RIC"], nominal["MidPrice"]))

    # Try batch rd.get_data for term fields
    has_terms = False
    try:
        term_fields = [
            "TR.MaturityDate", "TR.CouponRate", "TR.InterestPaymentFrequency",
            "TR.IssueDate", "TR.FirstCouponDate",
        ]
        terms = rd.get_data(universe=rics, fields=term_fields)
        if terms is not None and not terms.empty:
            terms = terms.rename(columns={
                "Instrument": "RIC",
                "Maturity Date": "MaturityDate", "TR.MaturityDate": "MaturityDate",
                "Coupon Rate": "CouponRate", "TR.CouponRate": "CouponRate",
                "Interest Payment Frequency": "CouponFrequency",
                "TR.InterestPaymentFrequency": "CouponFrequency",
                "Issue Date": "IssueDate", "TR.IssueDate": "IssueDate",
            })
            if "MaturityDate" in terms.columns and terms["MaturityDate"].notna().any():
                has_terms = True
                print(f"  rd.get_data returned term data for {terms['MaturityDate'].notna().sum()} bonds.")
    except Exception as e:
        print(f"  rd.get_data for terms failed: {e}")

    if not has_terms:
        print("  Falling back to bond IPA for term data...")
        ipa_fields = [
            "InstrumentCode", "EndDate", "InterestPaymentFrequency",
            "CouponRatePercent", "IssueDate", "FirstCouponDate",
        ]
        rows = []
        for i, ric in enumerate(rics):
            if i % 50 == 0:
                print(f"    IPA: {i + 1}/{len(rics)} ...")
            df_row = _ipa_get_data(ric, ipa_fields)
            if df_row is None:
                continue
            r = df_row.iloc[0]
            mat = r.get("EndDate")
            if mat is None or (isinstance(mat, float) and pd.isna(mat)):
                continue
            rows.append({
                "RIC": ric,
                "MaturityDate": mat,
                "CouponFrequency": _parse_frequency_to_int(r.get("InterestPaymentFrequency")),
                "CouponRate": r.get("CouponRatePercent"),
                "IssueDate": r.get("IssueDate"),
                "MidPrice": price_by_ric.get(ric),
            })
        if not rows:
            raise RuntimeError("IPA returned no term data for any coupon bond.")
        terms = pd.DataFrame(rows)
        print(f"  IPA returned data for {len(terms)} bonds.")

    # Normalise
    if "CouponFrequency" in terms.columns:
        terms["CouponFrequency"] = terms["CouponFrequency"].apply(_parse_frequency_to_int)
    for col in ["IssueDate", "MaturityDate", "FirstCouponDate"]:
        if col in terms.columns:
            terms[col] = pd.to_datetime(terms[col], errors="coerce").dt.date

    # Merge with raw prices (IPA path already has MidPrice, batch path needs it)
    if "MidPrice" not in terms.columns or terms["MidPrice"].isna().all():
        terms["MidPrice"] = terms["RIC"].map(price_by_ric)

    terms = terms.dropna(subset=["MaturityDate", "CouponRate", "MidPrice"]).copy()
    if "CouponFrequency" not in terms.columns:
        terms["CouponFrequency"] = 2

    valuation = date.today()
    terms["T_years"] = terms["MaturityDate"].apply(lambda d: _T_years(d, valuation))
    terms["Bucket"] = terms["T_years"].apply(_bucket)

    # Select bonds per bucket: sort by IssueDate (newest first), take top N
    def pick(g: pd.DataFrame) -> pd.DataFrame:
        if "IssueDate" in g.columns and g["IssueDate"].notna().any():
            sorted_g = g.sort_values("IssueDate", ascending=False)
        else:
            sorted_g = g.sort_values("T_years", ascending=False)
        if BONDS_PER_BUCKET is None:
            return sorted_g
        return sorted_g.head(BONDS_PER_BUCKET)

    selected = terms.groupby("Bucket", group_keys=False).apply(pick).sort_values("T_years")

    out = pd.DataFrame({
        "ric": selected["RIC"].values,
        "isin": selected.get("ISIN", pd.Series(dtype=str)).values if "ISIN" in selected.columns else None,
        "name": selected.get("Name", pd.Series(dtype=str)).values if "Name" in selected.columns else None,
        "maturity_date": selected["MaturityDate"].values,
        "T_years": selected["T_years"].values,
        "coupon_rate": selected["CouponRate"].values,
        "coupon_frequency": selected["CouponFrequency"].astype(int).values,
        "clean_price": selected["MidPrice"].values,
    })

    out_path = import_dir / f"us_treasury_input_{timestamp}.csv"
    out.to_csv(out_path, index=False)
    bonds_per_bucket_str = "all" if BONDS_PER_BUCKET is None else str(BONDS_PER_BUCKET)
    print(f"  Saved {len(out)} bonds ({bonds_per_bucket_str} per bucket) -> {out_path}")
    return out_path


# ============================================================================
# Step 3: Prepare US STRIPS
# ============================================================================

def _prepare_us_strips(raw_path: Path, timestamp: str, import_dir: Path) -> Path:
    """
    Fetch maturity dates for STRIPS via IPA, compute mid-price averages,
    save as us_strips_input_<timestamp>.csv (ric, maturity_date, T_years, mid_price).
    """
    print("\n=== Preparing US STRIPS ===")
    raw = pd.read_csv(raw_path).rename(columns=RENAME_RAW)

    # Use mid of bid/ask if MidPrice is missing
    if "MidPrice" not in raw.columns or raw["MidPrice"].isna().all():
        if "BidPrice" in raw.columns and "AskPrice" in raw.columns:
            raw["MidPrice"] = (raw["BidPrice"] + raw["AskPrice"]) / 2.0

    raw = raw.dropna(subset=["RIC"]).copy()
    # Keep rows that have at least one usable price
    raw = raw[raw["MidPrice"].notna() | raw["BidPrice"].notna()].copy()
    if "MidPrice" not in raw.columns or raw["MidPrice"].isna().any():
        raw["MidPrice"] = raw["MidPrice"].fillna(raw.get("BidPrice"))

    raw = raw.dropna(subset=["MidPrice"]).copy()
    if raw.empty:
        raise RuntimeError("No STRIPS with valid prices found.")

    rics = raw["RIC"].dropna().astype(str).unique().tolist()
    price_by_ric = dict(zip(raw["RIC"], raw["MidPrice"]))
    print(f"  {len(rics)} STRIPS with valid prices.")

    # Try batch first, fall back to IPA
    maturity_df = pd.DataFrame()
    try:
        batch = rd.get_data(universe=rics, fields=["TR.MaturityDate"])
        if batch is not None and not batch.empty:
            batch = batch.rename(columns={
                "Instrument": "RIC",
                "Maturity Date": "MaturityDate", "TR.MaturityDate": "MaturityDate",
            })
            if "MaturityDate" in batch.columns and batch["MaturityDate"].notna().any():
                batch["MaturityDate"] = pd.to_datetime(batch["MaturityDate"], errors="coerce").dt.date
                maturity_df = batch[["RIC", "MaturityDate"]].dropna()
                print(f"  rd.get_data returned maturities for {len(maturity_df)} STRIPS.")
    except Exception as e:
        print(f"  rd.get_data failed: {e}")

    if maturity_df.empty:
        print("  Falling back to bond IPA for STRIPS maturities...")
        rows = []
        for i, ric in enumerate(rics):
            if i % 50 == 0:
                print(f"    IPA: {i + 1}/{len(rics)} ...")
            df_row = _ipa_get_data(ric, ["InstrumentCode", "EndDate"])
            if df_row is None:
                continue
            mat = df_row.iloc[0].get("EndDate")
            if mat is None or (isinstance(mat, float) and pd.isna(mat)):
                continue
            rows.append({"RIC": ric, "MaturityDate": mat})
        if not rows:
            raise RuntimeError("IPA returned no maturities for any STRIP.")
        maturity_df = pd.DataFrame(rows)
        maturity_df["MaturityDate"] = pd.to_datetime(
            maturity_df["MaturityDate"], errors="coerce"
        ).dt.date
        maturity_df = maturity_df.dropna(subset=["MaturityDate"])
        print(f"  IPA returned maturities for {len(maturity_df)} STRIPS.")

    # Build output
    maturity_df["MidPrice"] = maturity_df["RIC"].map(price_by_ric)
    maturity_df = maturity_df.dropna(subset=["MidPrice"]).copy()

    valuation = date.today()
    maturity_df["T_years"] = maturity_df["MaturityDate"].apply(lambda d: _T_years(d, valuation))
    maturity_df = maturity_df[maturity_df["T_years"] > 0].sort_values("T_years")

    out = pd.DataFrame({
        "ric": maturity_df["RIC"].values,
        "maturity_date": maturity_df["MaturityDate"].values,
        "T_years": maturity_df["T_years"].values,
        "mid_price": maturity_df["MidPrice"].values,
    })

    out_path = import_dir / f"us_strips_input_{timestamp}.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved {len(out)} STRIPS -> {out_path}")
    return out_path


# ============================================================================
# Main
# ============================================================================

def main():
    global BONDS_PER_BUCKET

    parser = argparse.ArgumentParser(
        description="Download and prepare sovereign bond data from Refinitiv."
    )
    parser.add_argument(
        "--bonds-per-bucket",
        dest="bonds_per_bucket",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of bonds to select per maturity bucket. "
            "Overrides BONDS_PER_BUCKET in the script. "
            "Use 0 to keep all bonds (equivalent to BONDS_PER_BUCKET = None)."
        ),
    )
    args = parser.parse_args()

    # Override the global constant if the flag was provided
    if args.bonds_per_bucket is not None:
        BONDS_PER_BUCKET = None if args.bonds_per_bucket == 0 else args.bonds_per_bucket
        print(f"BONDS_PER_BUCKET overridden to: {BONDS_PER_BUCKET}")

    load_dotenv()
    api_key = os.getenv("REFINITIV_API_KEY")
    if not api_key:
        raise ValueError("REFINITIV_API_KEY not found in .env")

    # Full datetime stamp: YYYYMMDD_HHMMSS — unique per run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Each import gets its own folder: data/refinitiv/<timestamp>/
    import_dir = _make_import_dir(timestamp)
    print(f"Import folder: {import_dir}")

    print("Opening Refinitiv session...")
    rd.open_session(app_key=api_key)
    try:
        # Step 1: download raw universes into the import folder
        saved = _download_raw_universes(timestamp, import_dir)

        # Step 2: prepare US coupon bonds
        us_tsy_raw = saved.get("US_Treasury")
        if us_tsy_raw is None:
            candidates = sorted(import_dir.glob("US_Treasury_*.csv"))
            if candidates:
                us_tsy_raw = candidates[-1]
        if us_tsy_raw:
            _prepare_us_treasuries(us_tsy_raw, timestamp, import_dir)
        else:
            print("\nSkipping US Treasury preparation: no raw data available.")

        # Step 3: prepare US STRIPS
        us_zero_raw = saved.get("US_Zero")
        if us_zero_raw is None:
            candidates = sorted(import_dir.glob("US_Zero_*.csv"))
            if candidates:
                us_zero_raw = candidates[-1]
        if us_zero_raw:
            _prepare_us_strips(us_zero_raw, timestamp, import_dir)
        else:
            print("\nSkipping US STRIPS preparation: no raw data available.")

    finally:
        rd.close_session()

    print(f"\n=== All Refinitiv import & preparation complete — folder: {import_dir} ===")


if __name__ == "__main__":
    main()
