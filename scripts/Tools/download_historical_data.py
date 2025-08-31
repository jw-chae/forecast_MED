#!/usr/bin/env python3
"""
Historical data downloader for Zhejiang Health Commission
Downloads infectious disease reports from 2019-01 to 2022-01 for RAG system
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict
import re

def create_output_dir(base_path: str) -> str:
    """Create output directory if it doesn't exist"""
    output_dir = os.path.join(base_path, "reports", "evidence", "gov_reports")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def parse_date_from_title(title: str) -> str:
    """Extract date from title if possible"""
    # Look for patterns like 2020年1月, 2020-01, etc.
    date_patterns = [
        r'(\d{4})年(\d{1,2})月',  # 2020年1月
        r'(\d{4})-(\d{1,2})',    # 2020-01
        r'(\d{4})\.(\d{1,2})',   # 2020.01
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, title)
        if match:
            year, month = match.groups()
            return f"{year}-{month.zfill(2)}"
    return ""

def scrape_page(base_url: str, page_num: int) -> List[Dict]:
    """Scrape a single page of the website"""
    # Use the specific uid and pageNum format found in search results
    url = f"{base_url}?uid=4978845&pageNum={page_num}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            print(f"Failed to fetch page {page_num}: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = []
        
        # More specific selectors for this government website
        link_selectors = [
            'a[href*="art"]',  # Article links with 'art' in href
            'td a',            # Table cell links
            '.list a',         # List item links 
            'li a',            # Simple list links
            'a[title]',        # Links with title attribute
        ]
        
        for selector in link_selectors:
            links = soup.select(selector)
            if links:
                for link in links:
                    title = link.get_text(strip=True)
                    href = link.get('href', '')
                    
                    # More specific filtering for monthly infectious disease reports
                    if (any(keyword in title for keyword in ['传染病', '疫情', '法定']) and 
                        any(month_keyword in title for month_keyword in ['月', '年']) and
                        len(title) > 5 and  # Filter out very short titles
                        title != '疫情播报'):  # Exclude the generic "疫情播报" title
                        
                        # Extract date from title
                        date_str = parse_date_from_title(title)
                        
                        # Make absolute URL if relative
                        if href.startswith('/'):
                            href = 'https://wsjkw.zj.gov.cn' + href
                        elif not href.startswith('http'):
                            continue  # Skip invalid links
                        
                        # Avoid duplicates
                        if not any(article['url'] == href for article in articles):
                            articles.append({
                                'title': title,
                                'url': href,
                                'date': date_str,
                                'scraped_from_page': page_num,
                                'scraped_at': datetime.now().isoformat()
                            })
                
                if articles:  # If we found articles with this selector, use it
                    break
        
        print(f"Page {page_num}: Found {len(articles)} relevant articles")
        return articles
        
    except Exception as e:
        print(f"Error scraping page {page_num}: {e}")
        return []

def filter_by_date_range(articles: List[Dict], start_date: str, end_date: str) -> List[Dict]:
    """Filter articles by date range"""
    filtered = []
    
    for article in articles:
        if article['date']:
            try:
                article_date = datetime.strptime(article['date'], '%Y-%m')
                start_dt = datetime.strptime(start_date, '%Y-%m')
                end_dt = datetime.strptime(end_date, '%Y-%m')
                
                if start_dt <= article_date <= end_dt:
                    filtered.append(article)
            except ValueError:
                continue  # Skip articles with invalid dates
        else:
            # Include articles without clear dates for manual review
            filtered.append(article)
    
    return filtered

def download_historical_data(
    base_url: str = "https://wsjkw.zj.gov.cn/col/col1202112/",
    start_date: str = "2019-01",
    end_date: str = "2022-01",
    max_pages: int = 500,
    output_dir: str = None
) -> str:
    """
    Download historical infectious disease reports
    
    Args:
        base_url: Base URL of the government website
        start_date: Start date in YYYY-MM format
        end_date: End date in YYYY-MM format
        max_pages: Maximum pages to scrape
        output_dir: Output directory path
    
    Returns:
        Path to the saved JSON file
    """
    
    if output_dir is None:
        output_dir = create_output_dir(os.path.dirname(os.path.dirname(__file__)))
    
    print(f"Starting historical data download from {start_date} to {end_date}")
    print(f"Target URL: {base_url}")
    print(f"Max pages: {max_pages}")
    
    all_articles = []
    
    # Start from page 1 and go backwards in time
    for page_num in range(1, max_pages + 1):
        print(f"Scraping page {page_num}...")
        
        articles = scrape_page(base_url, page_num)
        
        if not articles:
            print(f"No articles found on page {page_num}, may have reached the end")
            # Try a few more pages in case of temporary issues
            if page_num > 10:  # Only stop if we've tried at least 10 pages
                break
        else:
            all_articles.extend(articles)
        
        # Be respectful to the server
        time.sleep(1)
        
        # Check if we have enough historical data
        if len(all_articles) > 0:
            # Sort by date and check if we've gone back far enough
            dated_articles = [a for a in all_articles if a['date']]
            if dated_articles:
                earliest_date = min(a['date'] for a in dated_articles if a['date'])
                if earliest_date and earliest_date < start_date:
                    print(f"Reached target start date. Earliest article: {earliest_date}")
                    break
    
    print(f"Total articles scraped: {len(all_articles)}")
    
    # Filter by date range
    filtered_articles = filter_by_date_range(all_articles, start_date, end_date)
    print(f"Articles in target date range: {len(filtered_articles)}")
    
    # Sort by date (newest first)
    filtered_articles.sort(key=lambda x: x['date'] or '9999-99', reverse=True)
    
    # Save to JSON file
    output_file = os.path.join(output_dir, f"historical_reports_{start_date}_to_{end_date}.json")
    
    metadata = {
        'download_date': datetime.now().isoformat(),
        'date_range': {'start': start_date, 'end': end_date},
        'total_articles': len(filtered_articles),
        'source_url': base_url,
        'articles': filtered_articles
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Historical data saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    # Download data from 2019-01 to 2022-01
    output_file = download_historical_data(
        start_date="2019-01",
        end_date="2022-01",
        max_pages=300  # Adjust based on website structure
    )
    
    print(f"Download completed. Data saved to: {output_file}")
