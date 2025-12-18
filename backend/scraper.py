import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin
import pandas as pd
from tqdm import tqdm
import os
import re


class SHLCatalogScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/products/product-catalog/"
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        self.products = []

    def get_page(self, url):
        """Fetch a page with retry logic"""
        for attempt in range(3):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except Exception as e:
                print(f"[Retry {attempt+1}] Failed: {e}")
                time.sleep(2)
        return None

    def extract_product_basic_info(self, row):
        """Extract product info from catalog row"""
        product = {}

        cells = row.find_all("td")
        if len(cells) < 4:
            return None

        link_tag = cells[0].find("a")
        if not link_tag:
            return None

        product["name"] = link_tag.text.strip()
        product["url"] = urljoin(self.base_url, link_tag["href"])

        product["remote_testing"] = bool(cells[1].text.strip())
        product["adaptive_irt"] = bool(cells[2].text.strip())

        test_types_text = cells[3].text.strip()
        product["test_types"] = test_types_text
        product["test_types_list"] = re.findall(r"[A-Z]", test_types_text)

        product["category"] = "Individual Test Solutions"

        return product

    def scrape_catalog_page(self, start):
        """Scrape a single page of Individual Test Solutions"""
        url = f"{self.base_url}?start={start}&type=1"
        soup = self.get_page(url)

        if not soup:
            return []

        products = []
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")[1:]
            for row in rows:
                product = self.extract_product_basic_info(row)
                if product:
                    products.append(product)

        return products

    def scrape_all_products(self):
        """Scrape ONLY Individual Test Solutions"""
        print("🚀 Scraping ONLY Individual Test Solutions")

        TOTAL_PAGES = 32
        ITEMS_PER_PAGE = 12

        all_products = []

        for page in tqdm(range(TOTAL_PAGES), desc="Pages"):
            start = page * ITEMS_PER_PAGE
            page_products = self.scrape_catalog_page(start)

            all_products.extend(page_products)
            time.sleep(1)

        self.products = all_products
        print(f"\n✅ Total Individual Tests scraped: {len(all_products)}")
        return all_products

    def save_outputs(self, base_path="backend/data"):
        """Save scraped data"""
        os.makedirs(base_path, exist_ok=True)

        json_path = os.path.join(base_path, "shl_individual_tests.json")
        csv_path = os.path.join(base_path, "shl_individual_tests.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.products, f, indent=2, ensure_ascii=False)

        pd.DataFrame(self.products).to_csv(
            csv_path, index=False, encoding="utf-8"
        )

        print(f"💾 Saved JSON → {json_path}")
        print(f"💾 Saved CSV  → {csv_path}")

    def print_summary(self):
        """Print summary"""
        print("\n📊 SCRAPING SUMMARY")
        print("=" * 60)
        print(f"Total Individual Tests: {len(self.products)}")

        test_type_counts = {}
        for p in self.products:
            for t in p.get("test_types_list", []):
                test_type_counts[t] = test_type_counts.get(t, 0) + 1

        print("\nBy Test Type:")
        for k in sorted(test_type_counts):
            print(f"  • {k}: {test_type_counts[k]}")


def main():
    print("=" * 70)
    print("SHL INDIVIDUAL TEST SOLUTION SCRAPER")
    print("=" * 70)

    scraper = SHLCatalogScraper()
    scraper.scrape_all_products()
    scraper.save_outputs()
    scraper.print_summary()

    print("\n✅ SCRAPING COMPLETE")
    print("➡️ Next step: data_enricher.py")


if __name__ == "__main__":
    main()
