"""
Collects sneaker listings from Jiji (Lagos) and Jumia Nigeria and saves them to sneakers.csv.

Each run adds new listings and refreshes ones already seen (listings are keyed by Link,
so nothing is duplicated). Every listing gets a canonical brand label from its title.

Use responsibly: keep the delay between requests, and respect each site's terms of use.
"""

import csv
import json
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------- SETTINGS ----------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "sneakers.csv"
CSV_COLUMNS = ["Date", "Title", "Price", "Brand", "Link"]

HEADERS = {"User-Agent": "SneakerScraper/1.0 (learning project)"}
REQUEST_TIMEOUT = 15      # seconds to wait before giving up on a request
PAUSE_BETWEEN_REQUESTS = 1

# ---------- BRAND LABELLING ----------
# Order matters: the first brand found in a title wins.
BRANDS = [
    "nike", "jordan", "adidas", "puma", "gucci", "louis vuitton", "lv", "hermes",
    "prada", "alexander mcqueen", "alexander macqueen", "balenciaga", "new balance",
    "vans", "asics", "ascis", "reebok", "dolce", "gabbana", "docle gabbana", "channel",
    "chanel", "boss", "marni", "dior", "rick owen", "hoka", "burberry", "ferragamo",
    "lacoste", "onitsuka", "zara", "levi", "skechers", "tommy jeans", "saucony",
    "converse", "maison margiela", "timberland", "tommy hilfiger", "givenchy",
    "santoni", "naked wolfe", "bape", "amiri", "sketchers",
]

# Spelling variants and sub-brands collapsed into one canonical label
BRAND_ALIASES = {
    "jordan": "nike",
    "docle gabbana": "dolce",
    "gabbana": "dolce",
    "channel": "chanel",
    "lv": "louis vuitton",
    "rick owen": "rick owens",
    "alexander macqueen": "alexander mcqueen",
    "sketchers": "skechers",
    "ascis": "asics",
}

# Model names that identify a brand when the brand itself is not in the title
PRODUCT_LINES = {
    "air max": "nike",
    "airforce": "nike",
    "air force": "nike",
}

# Must match as a complete word. Without this, "boss" matched "Bosses" and "Embossed".
WHOLE_WORD_BRANDS = {"lv", "boss"}


def get_brand(title):
    """Return the canonical brand named in a title, or "Other"."""
    title = title.lower()

    for brand in BRANDS:
        escaped = re.escape(brand)
        # Short or ambiguous names need a boundary on both sides; others only at the start
        pattern = rf"\b{escaped}\b" if (len(brand) <= 3 or brand in WHOLE_WORD_BRANDS) else rf"\b{escaped}"
        if re.search(pattern, title):
            return BRAND_ALIASES.get(brand, brand)

    for line, brand in PRODUCT_LINES.items():
        if re.search(rf"\b{re.escape(line)}", title):
            return brand

    return "Other"


# ---------- SCRAPERS ----------
class Scraper:
    """Base class. Each site gets a subclass that knows how to fetch its listings."""

    def __init__(self, name):
        self.name = name

    def fetch(self):
        raise NotImplementedError("Subclasses must implement fetch()")


class JijiScraper(Scraper):
    START_URL = "https://jiji.ng/api_web/v1/listing?page=1&query=Sneaker+in+lagos&init_page=true&webp=false"

    def __init__(self, num_batches=5):
        super().__init__("jiji")
        self.num_batches = num_batches

    def fetch(self):
        """Walk through Jiji's paginated API and return a list of listing dicts."""
        today = str(date.today())
        url = self.START_URL
        listings = []

        for _ in range(self.num_batches):
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            for advert in data["adverts_list"]["adverts"]:
                listings.append({
                    "Date": today,
                    "Title": " ".join(advert["title"].split()),     # collapse stray whitespace
                    "Price": float(advert["price_obj"]["value"]),
                    "Link": "https://jiji.ng" + advert["url"].split("?")[0],
                })

            url = data["next_url"]
            time.sleep(PAUSE_BETWEEN_REQUESTS)

        return listings


class JumiaScraper(Scraper):
    CATEGORY_URLS = [
        "https://www.jumia.com.ng/men-sneakers/",
        "https://www.jumia.com.ng/womens-sneakers/",
    ]

    def __init__(self, num_pages=10):
        super().__init__("jumia")
        self.num_pages = num_pages

    def fetch(self):
        """Fetch every page of every category and return a list of listing dicts."""
        listings = []
        for base_url in self.CATEGORY_URLS:
            for page in range(1, self.num_pages + 1):
                print(f"Fetching page {page} of {base_url}")
                soup = self.get_soup(f"{base_url}?page={page}")
                listings += self.build_listings(self.get_products(soup))
                time.sleep(PAUSE_BETWEEN_REQUESTS)
        return listings

    def get_soup(self, url):
        """Download a page, retrying up to 3 times. Returns an empty page if all attempts fail."""
        for attempt in range(1, 4):
            try:
                response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                return BeautifulSoup(response.text, "html.parser")
            except requests.exceptions.RequestException as error:
                print(f"WARNING: request failed (attempt {attempt}/3): {error}")
                time.sleep(2)
        print("ERROR: giving up on this page after 3 failed attempts")
        return BeautifulSoup("", "html.parser")

    def get_products(self, soup):
        """Pull the product list out of the JSON that Jumia embeds in the page as window.__STORE__."""
        html = str(soup)
        marker = html.find("window.__STORE__=")
        if marker == -1:
            print("WARNING: window.__STORE__= not found on this page, skipping")
            return []

        # Find the matching closing brace by counting { and }
        json_start = html.find("{", marker)
        depth = 0
        json_end = None
        for i in range(json_start, len(html)):
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        if json_end is None:
            print("WARNING: could not find the end of the JSON data, skipping")
            return []

        products = json.loads(html[json_start:json_end]).get("products", [])
        if not products:
            print("WARNING: no products found on this page")
        return products

    def build_listings(self, products):
        today = str(date.today())
        return [{
            "Date": today,
            "Title": product["name"].split(",")[0],
            "Price": float(product["prices"]["rawPrice"]),
            "Brand": product["brand"],          # Jumia supplies its own brand field
            "Link": "https://www.jumia.com.ng" + product["url"],
        } for product in products]


# ---------- SAVING ----------
def load_existing(path):
    """Read the CSV into a dict keyed by Link, so repeat listings overwrite instead of duplicating."""
    existing = {}
    if not path.exists():
        return existing

    with open(path, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            existing[row["Link"]] = {
                "Date": row.get("Date", "Unknown"),
                "Title": row["Title"],
                "Price": float(row["Price"]),
                "Brand": row.get("Brand", "Other"),
                "Link": row["Link"],
            }
    return existing


def save(path, listings):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(listings.values())


def main():
    jiji_listings = JijiScraper(num_batches=5).fetch()
    jumia_listings = JumiaScraper(num_pages=10).fetch()

    listings = load_existing(CSV_PATH)

    for item in jiji_listings:
        item["Brand"] = get_brand(item["Title"])
        listings[item["Link"]] = item

    for item in jumia_listings:
        brand = get_brand(item["Brand"])           # try Jumia's brand field first
        item["Brand"] = brand if brand != "Other" else get_brand(item["Title"])
        listings[item["Link"]] = item

    save(CSV_PATH, listings)
    print(f"Saved {len(listings)} listings to {CSV_PATH}")


if __name__ == "__main__":
    main()
