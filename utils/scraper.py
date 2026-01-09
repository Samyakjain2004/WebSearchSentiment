import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config
import asyncio

def fetch_articles_from_source(source_name, source_url, max_articles=10):
    # Configure newspaper with a custom user-agent
    config = Config()
    config.browser_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    config.request_timeout = 10
    config.number_threads = 1

    articles = []
    try:
        resp = requests.get(source_url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        links = list({a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('http')})
        for link in links[:max_articles]:  # Respect total max_articles
            for attempt in range(3):  # Retry up to 3 times
                try:
                    art = Article(link, config=config)
                    art.download()
                    art.parse()
                    articles.append({
                        'source': source_name,
                        'url': link,
                        'title': art.title,
                        'content': art.text
                    })
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        print(f"Error processing article {link}: {str(e)}")
                    asyncio.sleep(1)  # Synchronous sleep (for simplicity in non-async function)
    except Exception as e:
        print(f"Error processing website {source_name}: {str(e)}")
    return articles