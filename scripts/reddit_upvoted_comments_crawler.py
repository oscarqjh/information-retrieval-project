import requests
from bs4 import BeautifulSoup
import json
import time
import random
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# List of subreddits to crawl
SUBREDDITS = ["aiwars", "aiart", "ArtistLounge", "ArtistHate"]
TOTAL_RECORDS_NEEDED = 100
MIN_SCORE_THRESHOLD = 5  # Only save comments with this many upvotes
OUTPUT_FILE = "reddit_data.jsonl"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SC4021_Project_Scraper_v3'}

def get_robust_session():
    session = requests.Session()
    retry = Retry(
        total=5, 
        backoff_factor=2, 
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def parse_score(score_text):
    """Converts Reddit score text (e.g., '1.2k', '15 points') to integer."""
    if not score_text or "hidden" in score_text.lower():
        return 0
    try:
        score_text = score_text.lower()
        # Extract numeric parts
        match = re.search(r'(-?\d+\.?\d*)', score_text)
        if not match:
            return 0
        
        val = float(match.group(1))
        if 'k' in score_text:
            val *= 1000
        return int(val)
    except:
        return 0

def scrape_comments_as_records(session, post_url, post_title, sub_name):
    """Visits a post and returns each qualified comment as a standalone record."""
    records = []
    try:
        url = post_url.replace("www.reddit.com", "old.reddit.com")
        if not url.startswith('http'): 
            url = "https://old.reddit.com" + url
            
        resp = session.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Target individual comment entries
        comment_entries = soup.select('div.commentarea > div.sitetable > div.thing.comment')
        
        for entry in comment_entries:
            # Check score (i.e. upvotes)
            score_el = entry.select_one('span.score')
            score_val = parse_score(score_el.get_text()) if score_el else 0
            
            if score_val < MIN_SCORE_THRESHOLD:
                continue

            # Get the comment body
            body = entry.select_one('div.usertext-body')
            if body:
                text = body.get_text(strip=True)
                # Filter out standard noise and very short comments
                if len(text) > 20 and text != "[deleted]" and text != "[removed]":
                    records.append({
                        "subreddit": sub_name,
                        "parent_title": post_title,
                        "text": text,
                        "score": score_val,
                        "type": "comment"
                    })
        return records
    except Exception as e:
        print(f"Error on {post_url}: {e}")
        return []

def run_sorted_crawler(subreddit_list, sort_type="top", time_filter="all"):
    session = get_robust_session() 
    total_count = 0
    
    

    for sub in subreddit_list:
        if total_count >= TOTAL_RECORDS_NEEDED: 
            break
        
        # Determine starting URL based on sort
        if sort_type == "top":
            current_url = f"https://old.reddit.com/r/{sub}/top/?sort=top&t={time_filter}"
        else:
            current_url = f"https://old.reddit.com/r/{sub}/{sort_type}/"
            
        print(f"\n--- Gathering {sort_type} posts from r/{sub} ---")
        
        while total_count < TOTAL_RECORDS_NEEDED:
            try:
                res = session.get(current_url, headers=HEADERS, timeout=10)
                if res.status_code != 200:
                    print(f"HTTP Error {res.status_code} on listing page. Moving to next sub.")
                    break
                    
                soup = BeautifulSoup(res.text, 'html.parser')
                posts = soup.find_all('div', class_='thing')
                
                if not posts:
                    print("No more posts found.")
                    break
                
                for post in posts:
                    title_link = post.find('a', class_='title')
                    if not title_link: 
                        continue
                    
                    # Deep scrape -- get comments within this post
                    new_records = scrape_comments_as_records(session, title_link['href'], title_link.text, sub)
                    
                    if new_records:
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            for record in new_records:
                                f.write(json.dumps(record) + "\n")
                                total_count += 1
                        
                        print(f"Total: {total_count} | Just added {len(new_records)} from '{title_link.text[:30]}...'")
                    
                    time.sleep(random.uniform(1.5, 3.0))
                    
                    if total_count >= TOTAL_RECORDS_NEEDED: 
                        break
                
                # Pagination
                next_btn = soup.find('span', class_='next-button')
                if next_btn and next_btn.find('a'):
                    current_url = next_btn.find('a')['href']
                else: 
                    break
                    
            except Exception as e:
                print(f"Loop error: {e}")
                time.sleep(5)
                continue

if __name__ == "__main__":
    # Can change "top" with "all", "year", "month" or "best"/"hot"
    run_sorted_crawler(SUBREDDITS, sort_type="best")