"""
Sneaker price prediction for Lagos listings (Jiji + Jumia).

Predicts a listing's price from its brand and title words using linear
regression. Features are added one round at a time and every round is scored
with 5-fold cross-validation, so a feature is only kept if it beats the noise.

Run scraper.py first to create sneakers.csv. The repo only includes a 50-row sample.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

# ---------- SETTINGS ----------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "sneakers.csv"      # full data, created by scraper.py (not stored in the repo)
SAMPLE_PATH = BASE_DIR / "sneakers_sample.csv"   # 50-row preview that IS in the repo

MIN_BRAND_LISTINGS = 10                   # brands with fewer rows are too thin to learn from
N_FOLDS = 5
RANDOM_STATE = 42                         # same shuffling every run, so results are repeatable

TIER_WORDS = ["original", "quality", "luxury", "premium"]       # matched anywhere in the title
MODEL_NAMES = ["air force", "air max", "campus", "chuck", "gel"]  # Round 4: added no proven gain
EXTRA_WORDS = ["women", "free", "designer", "men", "new"]       # matched as whole words only


# ---------- DATA ----------
def load_data(path):
    """Load the CSV, keep labelled brands with enough listings."""
    df = pd.read_csv(path)
    df["source"] = df["Link"].apply(lambda link: "jiji" if "jiji.ng" in link else "jumia")

    df = df[df["Brand"] != "Other"]                      # no usable brand signal
    counts = df["Brand"].value_counts()
    df = df[df["Brand"].isin(counts[counts >= MIN_BRAND_LISTINGS].index)]
    return df.copy()


# ---------- FEATURES ----------
def keyword_columns(titles, words, whole_word=False):
    """One 0/1 column per word: 1 if the word appears in the title.

    whole_word=False: "men" would also match inside "women".
    whole_word=True : only the exact word counts.
    """
    titles = titles.str.lower()
    word_sets = titles.str.split().apply(set)
    columns = {}
    for word in words:
        if whole_word:
            hits = word_sets.apply(lambda words_in_title: word in words_in_title)
        else:
            hits = titles.str.contains(word, regex=False)
        columns["kw_" + word.replace(" ", "_")] = hits.astype(int)
    return pd.DataFrame(columns, index=titles.index)


def build_feature_sets(df):
    """Return {round name: feature table}. Each round adds features to an earlier one."""
    # drop_first=True: one brand becomes the reference (all zeros), avoiding redundant columns
    brands = pd.get_dummies(df["Brand"], drop_first=True, dtype=int)

    tier = keyword_columns(df["Title"], TIER_WORDS)
    models = keyword_columns(df["Title"], MODEL_NAMES)
    extra = keyword_columns(df["Title"], EXTRA_WORDS, whole_word=True)
    available = keyword_columns(df["Title"], ["available"], whole_word=True)
    is_jumia = (df["source"] == "jumia").astype(int).rename("is_jumia")

    round3 = pd.concat([brands, tier], axis=1)
    round5 = pd.concat([round3, extra], axis=1)

    return {
        "Round 1: brand only": brands,
        "Round 2: + source": pd.concat([brands, is_jumia], axis=1),
        "Round 3: + tier words": round3,
        "Round 4: + model names": pd.concat([round3, models], axis=1),
        "Round 5: + more words": round5,
        "Round 6: + 'available'": pd.concat([round5, available], axis=1),
    }


# ---------- EVALUATION ----------
def cross_validated_mae(model, X, y):
    """MAE on each of the 5 folds. Same folds every call, so rounds compare fairly."""
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # sklearn returns MAE as a negative number (higher = better), so flip the sign
    return -cross_val_score(model, X, y, cv=folds, scoring="neg_mean_absolute_error")


def report(df, feature_sets):
    """Print the cross-validated MAE of the naive baseline and every round."""
    y = df["Price"]

    baseline = cross_validated_mae(DummyRegressor(strategy="median"), feature_sets["Round 1: brand only"], y)
    print(f"Naive baseline (always guess the median): MAE N{baseline.mean():,.0f} (+/- N{baseline.std():,.0f})")

    for name, X in feature_sets.items():
        scores = cross_validated_mae(LinearRegression(), X, y)
        print(f"{name}: MAE N{scores.mean():,.0f} (+/- N{scores.std():,.0f})")


# ---------- FINAL MODEL ----------
def estimate_price(model, feature_columns, reference_brand, brand, title):
    """Estimate the price of one listing with the final (Round 5) model."""
    brand = brand.lower()
    if brand not in feature_columns and brand != reference_brand:
        print(f"Warning: '{brand}' is not a brand the model was trained on.")

    row = pd.DataFrame(0, index=[0], columns=feature_columns)   # all zeros = the reference brand
    if brand in row.columns:
        row[brand] = 1

    titles = pd.Series([title])
    keywords = pd.concat([
        keyword_columns(titles, TIER_WORDS),
        keyword_columns(titles, EXTRA_WORDS, whole_word=True),
    ], axis=1)
    for column in keywords.columns:
        row[column] = keywords[column].iloc[0]

    return model.predict(row)[0]


def main():
    if not CSV_PATH.exists():
        print("sneakers.csv not found.")
        print("The repo only ships a 50-row preview (sneakers_sample.csv), which is far too small to train on.")
        print("Run scraper.py first to build the full dataset, then run this script again.")
        return

    df = load_data(CSV_PATH)
    print(f"{len(df)} listings, {df['Brand'].nunique()} brands\n")

    feature_sets = build_feature_sets(df)
    report(df, feature_sets)

    # Fit the final model on all the data (the testing is already done above)
    X_final = feature_sets["Round 5: + more words"]
    final_model = LinearRegression().fit(X_final, df["Price"])
    reference_brand = sorted(df["Brand"].unique())[0]     # the brand with all-zero columns

    print("\nExample estimates (typical error is about N27,000, so treat these as ranges):")
    for brand, title in [("nike", "Original Nike Air Force 1 Sneakers"),
                         ("gucci", "Luxury Men Gucci Sneakers")]:
        price = estimate_price(final_model, X_final.columns, reference_brand, brand, title)
        print(f"  {title}: N{price:,.0f}")


if __name__ == "__main__":
    main()
