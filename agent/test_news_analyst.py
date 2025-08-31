import os
from datetime import datetime, timedelta
import pytz
from serpapi import GoogleSearch
from dotenv import load_dotenv
import re
from dateutil import parser, tz

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

def parse_relative_date(date_str, ist_tz):
    """Parse relative date strings like '2 hours ago', '3 days ago', etc."""
    try:
        # Extract number and unit from relative time
        match = re.search(r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', date_str.lower())
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            
            # Calculate the date based on the relative time
            now = datetime.now(ist_tz)
            if unit == 'second':
                return now - timedelta(seconds=num)
            elif unit == 'minute':
                return now - timedelta(minutes=num)
            elif unit == 'hour':
                return now - timedelta(hours=num)
            elif unit == 'day':
                return now - timedelta(days=num)
            elif unit == 'week':
                return now - timedelta(weeks=num)
            elif unit == 'month':
                return now - timedelta(days=num*30)  # Approximation
            elif unit == 'year':
                return now - timedelta(days=num*365)  # Approximation
    except Exception:
        pass
    
    return None

def fetch_news(query, limit=5, days_limit=120):
    """
    Fetch recent company-specific news using SerpAPI Google News.
    Returns list of dicts: {"title": str, "link": str, "published": str}.
    Converts timestamps to IST and filters news within `days_limit` days.
    """
    if not SERPAPI_KEY:
        print("⚠️ Missing SERPAPI_API_KEY")
        return []
    
    ist_tz = pytz.timezone("Asia/Kolkata")
    cutoff_date = datetime.now(ist_tz) - timedelta(days=days_limit)
    
    params = {
        "engine": "google_news",
        "q": f"{query} stock OR shares OR finance",
        "api_key": SERPAPI_KEY,
        "num": limit * 2,  # fetch extra to filter by date
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        news_results = results.get("news_results") or []
        
        structured_news = []
        for article in news_results:
            title = article.get("title", "No Title")
            link = article.get("link", "")
            published_str = article.get("date", "")
            
            dt_ist = None
            
            # Try to parse the published date
            if published_str:
                # First, try to parse as relative date
                dt_ist = parse_relative_date(published_str, ist_tz)
                
                # If relative parsing failed, try other formats
                if dt_ist is None:
                    # Try to parse as UNIX timestamp
                    try:
                        # Check if it's a number (UNIX timestamp)
                        if str(published_str).isdigit():
                            ts = int(float(published_str))
                            dt_utc = datetime.fromtimestamp(ts, tz=pytz.utc)
                            dt_ist = dt_utc.astimezone(ist_tz)
                    except Exception:
                        pass
                
                # If still not parsed, try with dateutil.parser
                if dt_ist is None:
                    try:
                        # Try to parse with dateutil.parser
                        dt = parser.parse(published_str, fuzzy=True)
                        
                        # If the datetime is naive (no timezone), assume UTC
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=tz.gettz('UTC'))
                        
                        # Convert to IST
                        dt_ist = dt.astimezone(ist_tz)
                    except Exception as e:
                        print(f"Error parsing date '{published_str}': {e}")
            
            # If all parsing failed, use current time but log it
            if dt_ist is None:
                dt_ist = datetime.now(ist_tz)
                print(f"Warning: Could not parse date '{published_str}', using current time")
            
            # Check if the news is within the time limit
            if dt_ist < cutoff_date:
                continue  # discard old news
            
            structured_news.append({
                "title": title,
                "link": link,
                "published": dt_ist.strftime("%d %b %Y, %I:%M %p IST")
            })
            
            if len(structured_news) >= limit:
                break
        
        if not structured_news:
            return [{"title": f"No recent news found for {query} in last {days_limit} days.",
                     "link": "", "published": ""}]
        
        return structured_news
    except Exception as e:
        print(f"Error fetching news: {e}")
        return [{"title": f"Error fetching news: {e}", "link": "", "published": ""}]

# -------------------------
# Standalone test
# -------------------------
if __name__ == "__main__":
    for news in fetch_news("TCS", limit=10):
        print(f"{news['published']} – {news['title']}\n   🔗 {news['link']}\n")