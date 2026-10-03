"""
Exploratory analysis of the scraped sneaker listings.

Produces three things from sneakers.csv:
  1. Median price by brand (Jiji) as a chart
  2. Listings flagged as possible replicas (replica words and/or unusually low price)
  3. Median price over time, by source, as a chart

Uses sneakers.csv (from scraper.py), or the 50-row sample if that is missing. Charts are saved to a charts/ folder.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------- SETTINGS ----------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "sneakers.csv"                 # full data, created by scraper.py
SAMPLE_PATH = BASE_DIR / "sneakers_sample.csv"       # 50-row preview included in the repo
CHART_DIR = BASE_DIR / "charts"

MIN_LISTINGS_FOR_CHART = 3

REPLICA_WORDS = ["replica", "aaa", "1:1", "mirror", "high quality",
                 "original quality", "premium quality"]

LUXURY_BRANDS = ["gucci", "balenciaga", "prada", "dolce", "givenchy", "chanel", "hermes",
                 "dior", "marni", "burberry", "ferragamo", "alexander mcqueen", "amiri",
                 "louis vuitton", "maison margiela"]

# A listing is "low price" if it costs less than this share of its brand's median.
# Luxury brands vary more in price, so their cutoff is looser.
LUXURY_CUTOFF = 0.6
EVERYDAY_CUTOFF = 0.4


# ---------- LOADING ----------
def load_data(path):
    df = pd.read_csv(path)
    df["source"] = df["Link"].apply(lambda link: "jiji" if "jiji.ng" in link else "jumia")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------- FLAGS ----------
def price_cutoff(brand):
    return LUXURY_CUTOFF if brand in LUXURY_BRANDS else EVERYDAY_CUTOFF


def add_flags(df):
    """Add replica-word and low-price flags, plus a combined Flag column."""
    pattern = "|".join(re.escape(word) for word in REPLICA_WORDS)
    df["replicaword"] = df["Title"].str.contains(pattern, case=False, regex=True)

    # Compare each price with the median price of its own brand
    brand_median = df.groupby("Brand")["Price"].transform("median")
    df["PriceRatio"] = df["Price"] / brand_median
    df["LowPrice"] = df["PriceRatio"] < df["Brand"].apply(price_cutoff)

    df["Flag"] = ""
    df.loc[df["replicaword"] & df["LowPrice"], "Flag"] = "replica_word+low_price"
    df.loc[df["replicaword"] & ~df["LowPrice"], "Flag"] = "replica_word"
    df.loc[~df["replicaword"] & df["LowPrice"], "Flag"] = "low_price"
    return df


# ---------- CHARTS ----------
def plot_brand_medians(df, save_path):
    """Horizontal bar chart of median price per brand on Jiji (brands with enough listings only).

    Horizontal bars keep long brand names readable; vertical bars made the labels overlap.
    """
    summary = df[df["source"] == "jiji"].groupby("Brand")["Price"].agg(["median", "count"])
    summary = summary[(summary["count"] >= MIN_LISTINGS_FOR_CHART) & (summary.index != "Other")]
    summary = summary.sort_values("median")

    labels = [f"{brand} (n={count})" for brand, count in zip(summary.index, summary["count"])]

    fig, ax = plt.subplots(figsize=(8, 0.28 * len(summary) + 1.5))   # taller chart for more brands
    ax.barh(labels, summary["median"])
    ax.set_title("Median Sneaker Price by Brand (Jiji, Lagos)")
    ax.set_xlabel("Median price (₦)")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")   # 100000 -> 100,000
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_price_trend(df, save_path):
    """Line chart of the daily median price, one line per source."""
    daily = df.groupby(["Date", "source"])["Price"].median().reset_index()

    fig, ax = plt.subplots()
    for source, group in daily.groupby("source"):
        ax.plot(group["Date"], group["Price"], marker="o", label=source)
    ax.set_title("Median Sneaker Price Over Time by Source")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₦)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------- MAIN ----------
def main():
    path = CSV_PATH
    if not CSV_PATH.exists():
        print("sneakers.csv not found, using the 50-row sample. Run scraper.py for real results.\n")
        path = SAMPLE_PATH

    df = add_flags(load_data(path))
    CHART_DIR.mkdir(exist_ok=True)

    print("Listings per source:")
    print(df["source"].value_counts(), "\n")

    print("Flagged listings:")
    print(df["Flag"].replace("", "not flagged").value_counts(), "\n")

    # Flags only mean something for labelled brands (the "Other" group has no real median)
    suspicious = df[df["LowPrice"] & (df["Brand"] != "Other")]
    print("Lowest-priced suspicious listings:")
    print(suspicious.sort_values("PriceRatio")[["Title", "Brand", "Price", "PriceRatio"]].head(10).to_string())

    plot_brand_medians(df, CHART_DIR / "brand_chart.png")
    plot_price_trend(df, CHART_DIR / "price_trend_chart.png")
    print(f"\nCharts saved to {CHART_DIR}")


if __name__ == "__main__":
    main()
