import pandas as pd
from pathlib import Path

FACE_VALUE = 100.0

synthetic_bonds = [
    # maturity_years, coupon_rate (decimal), price (% of face), freq (payments/year)
    {"maturity_years": 2.0, "coupon_rate": 0.040, "price": 100.00, "freq": 2},
    {"maturity_years": 3.0, "coupon_rate": 0.042, "price": 100.50, "freq": 2},
    {"maturity_years": 5.0, "coupon_rate": 0.045, "price": 101.20, "freq": 2},
]

bonds_df = pd.DataFrame(synthetic_bonds)
print(bonds_df)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
bonds_path = data_dir / "synthetic_treasury_notes.csv"
bonds_df.to_csv(bonds_path, index=False)
bonds_path