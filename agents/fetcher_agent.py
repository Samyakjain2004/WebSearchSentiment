from utils.scraper import fetch_articles_from_source

NEWS_SOURCES = [
    {"name": "Economic Times", "url": "https://economictimes.indiatimes.com/news/economy"},
    {"name": "Moneycontrol", "url": "https://www.moneycontrol.com/news/business/"},
    {"name": "Bloomberg", "url": "https://www.bloomberg.com/markets/economics"},
    {"name": "Google News", "url": "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZ4Y0dNU0FtVnVLQUFQAQ?oc=3&ceid=IN:en"},
    {"name": "Bing News", "url": "https://www.bing.com/news/search?q=economy"},
]

def fetch_news_articles():
    articles = []
    for source in NEWS_SOURCES:
        articles += fetch_articles_from_source(source["name"], source["url"], max_articles=10)
    return articles 