import argparse
import json
from pathlib import Path
import time
import sys

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

def setup_selenium_driver():
    """Initializes a headless Chrome browser using Selenium."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def crawl_with_selenium(driver, url: str) -> str:
    """Navigates to a URL and returns the page source."""
    driver.get(url)
    # Wait for dynamic content to load, adjust time if needed
    time.sleep(5) 
    return driver.page_source


def pre_crawl_official_reports(jsonl_path: str, output_dir: str):
    """
    Reads a JSONL file containing URLs of official reports, crawls each URL,
    and saves the raw HTML content to a local directory.
    """
    jsonl_p = Path(jsonl_path)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    reports = []
    with open(jsonl_p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                reports.append(json.loads(line))

    print(f"Found {len(reports)} reports to crawl from {jsonl_path}")
    
    driver = None
    try:
        print("Setting up Selenium WebDriver...")
        driver = setup_selenium_driver()
        print("WebDriver setup complete.")

        for i, report in enumerate(reports):
            month = report.get("month")
            url = report.get("url")

            if not month or not url:
                continue

            safe_month = month.replace("-", "_")
            target_file = out_p / f"{safe_month}_report.html"

            if target_file.exists():
                print(f"[{i+1}/{len(reports)}] Skipping {month}, already downloaded.")
                continue

            print(f"[{i+1}/{len(reports)}] Crawling {month} report from {url}...")
            try:
                raw_html = crawl_with_selenium(driver, url)
                if raw_html:
                    target_file.write_text(raw_html, encoding="utf-8")
                    print(f"  -> Saved to {target_file}")
                else:
                    print(f"  -> [WARN] No HTML content found for {url}")
                
            except Exception as e:
                print(f"  -> [ERROR] Failed to crawl {url}: {e}")
                error_file = out_p / f"{safe_month}_report.error.txt"
                error_file.write_text(str(e), encoding="utf-8")
    
    finally:
        if driver:
            driver.quit()
        print("\nCrawling process completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True, help="Path to the monthly stats JSONL file.")
    parser.add_argument("--output_dir", default="./reports/crawled_html", help="Directory to save the crawled HTML files.")
    args = parser.parse_args()
    
    pre_crawl_official_reports(args.jsonl_path, args.output_dir)

if __name__ == "__main__":
    main()
