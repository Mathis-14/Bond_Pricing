"""
Download sovereign bond data from Refinitiv for use in bootstrapping.

This script:
  - Opens a Refinitiv Data Platform session using REFINITIV_API_KEY from .env
  - Builds several sovereign bond universes (US + main Eurozone issuers)
  - Uses discovery.search + get_data to retrieve identifiers, terms, dates, and prices
  - Saves one CSV per bond type under data/refinitiv/<UniverseName>_YYYYMMDD.csv

Later, a separate module will transform these raw CSVs into a canonical
bootstrapping dataset and implement the actual curve bootstrapping logic.

Note: Refinitiv also exposes an IPA bond content layer via
refinitiv.data.content.bond.Definition and bond.PricingParameters with fields
like Price, YieldPercent, ZSpreadBp, MarketDataDate. For large universes we keep
this importer on discovery + get_data for efficiency and will use the IPA layer
in focused analytics examples documented in REFINITIV_README.md.
"""

import os
import datetime
from pathlib import Path
from dotenv import load_dotenv
import refinitiv.data as rd
import pandas as pd

# Load environment variables
load_dotenv()
API_KEY = os.getenv("REFINITIV_API_KEY")

if not API_KEY:
    raise ValueError("REFINITIV_API_KEY not found in .env")

# Configuration for Bond Universes
# Each universe has a Name and a Filter for the search
BOND_UNIVERSES = {
    # US nominal Treasuries (notes & bonds, excluding zeros;
    # inflation-linked issues will be filtered later in Python if needed)
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
    # German government curve: Bunds, Bobls, Schatz
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
    # French OAT curve
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
    # Italian BTP curve
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
    # US STRIPS / zeros
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
    # French zero-coupon STRIPS
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

# Fields to retrieve
# Focusing on Price, Terms, Dates, and Identifiers
FIELDS = [
    # Identifiers
    "ISIN",
    "RIC",
    "Ticker",
    # Metadata
    "TR.CommonName",
    "TR.FiCurrency",
    "TR.AssetStatus",
    "TR.IssuerCountry",
    # Pricing (Snapshot)
    "TR.BidPrice",
    "TR.AskPrice",
    "TR.MidPrice",
    "TR.Yield",
    # Terms for Bootstrapping
    "TR.CouponRate",
    "TR.InterestPaymentFrequency",  # Annual, Semi-Annual, etc.
    "TR.DayCountBasis",  # Actual/Actual, 30/360, etc.
    # Dates
    "TR.IssueDate",
    "TR.MaturityDate",
    "TR.FirstCouponDate",
]


def main():
    print("Opening Refinitiv Session...")
    try:
        rd.open_session(app_key=API_KEY)
        print("Session opened successfully.")
    except Exception as e:
        print(f"Failed to open session: {e}")
        return

    # Ensure output directory exists
    output_dir = Path("data/refinitiv")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d")

    for universe_name, config in BOND_UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        print(f"Description: {config['description']}")
        print(f"Filter: {config['filter']}")
        
        try:
            # 1. Search for instruments
            print(f"Searching for {universe_name}...")
            # We request a high top count to get 'all' available liquid bonds
            # Usually active sovereign curves have < 100-200 points per country, so 1000 is safe.
            search_response = rd.discovery.search(
                view=rd.discovery.Views.GOV_CORP_INSTRUMENTS,
                filter=config['filter'],
                select="RIC, DTSubjectName",
                top=2000 
            )
            
            if search_response is None or search_response.empty:
                print(f"No instruments found for {universe_name}.")
                continue

            # Clean RIC list: drop missing values and ensure pure string list
            rics = (
                search_response["RIC"]
                .dropna()
                .astype(str)
                .tolist()
            )

            if not rics:
                print(f"No valid RICs found for {universe_name} after cleaning.")
                continue

            print(f"Found {len(rics)} instruments.")
            
            # 2. Retrieve Data
            print(f"Retrieving data for {len(rics)} instruments...")
            
            # get_data allows batch retrieval
            # We might need to chunk if > 5000, but for < 2000 it should be fine.
            df = rd.get_data(
                universe=rics,
                fields=FIELDS
            )
            
            if df is None or df.empty:
                print(f"No data returned for {universe_name}.")
                continue
                
            # 3. Clean and Save
            # Drop rows where critical pricing or maturity is missing?
            # For now, we save raw data and let the bootstrapper handle cleaning.
            
            filename = f"{universe_name}_{timestamp}.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} rows to {filepath}")
            
        except Exception as e:
            print(f"Error processing {universe_name}: {e}")

    print("\nAll processing complete.")
    rd.close_session()

if __name__ == "__main__":
    main()
