import asyncio
import json
import subprocess
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
import openai
from datetime import datetime

load_dotenv()

# Configure OpenAI
if os.getenv("AZURE_OPENAI_API_KEY"):
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    deployment_name = os.getenv('AZURE_OPENAI_MODEL_NAME')
else:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def call_openai(prompt: str, model: str = None) -> str:
    """Call OpenAI API with proper error handling"""
    try:
        if os.getenv("AZURE_OPENAI_API_KEY"):
            # Azure OpenAI
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512
            )
        else:
            # Regular OpenAI
            model = model or "gpt-3.5-turbo"
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512
            )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

class MCPClient:
    """Simplified MCP Client that uses direct function calls instead of MCP servers"""

    def __init__(self):
        self.servers_initialized = True  # Always true since we're using direct calls

    async def initialize_servers(self):
        """Initialize servers (no-op for simplified client)"""
        print("Using simplified MCP client with direct function calls")
        return True

    async def fetch_news_from_rss(self, feed_urls: List[str], max_articles: int = 10, company_ratio: float = 0.7) -> Dict[str, Any]:
        """Fetch news from RSS feeds respecting company/economic ratio strictly"""
        try:
            import feedparser
            from newspaper import Article

            articles = []
            company_articles = []
            economic_articles = []
            errors = []

            # Calculate target counts
            target_company = int(max_articles * company_ratio)
            target_economic = max_articles - target_company

            total_feeds = len(feed_urls)
            if total_feeds == 0:
                return {"success": False, "error": "No RSS feeds provided", "articles": [], "errors": []}

            per_feed_limit = max(1, max_articles // total_feeds)  # Initial limit per feed

            # Round-robin fetching
            feed_indices = {url: 0 for url in feed_urls}  # Track current article index per feed
            feeds = {url: feedparser.parse(url) for url in feed_urls}

            while len(articles) < max_articles:
                all_feeds_exhausted = True
                for feed_url in feed_urls:
                    feed = feeds[feed_url]
                    current_index = feed_indices[feed_url]

                    if current_index >= len(feed.entries):
                        continue  # This feed is exhausted

                    all_feeds_exhausted = False
                    entry = feed.entries[current_index]
                    feed_indices[feed_url] += 1

                    try:
                        article = Article(entry.link)
                        article.download()
                        article.parse()

                        # Classify article to determine its type
                        classification_result = await self.classify_article(
                            article.title or entry.title,
                            article.text
                        )
                        classification = classification_result.get("analysis", {}).get("classification", "unknown")

                        # Prepare article data
                        article_data = {
                            'url': entry.link,
                            'title': article.title or entry.title,
                            'content': article.text,
                            'source': feed_url,
                            'published_date': entry.get('published', ''),
                            'summary': entry.get('summary', '')
                        }

                        # Add to appropriate bucket only if not full
                        if classification == "company" and len(company_articles) < target_company:
                            company_articles.append(article_data)
                            articles.append(article_data)
                        elif classification == "economic" and len(economic_articles) < target_economic:
                            economic_articles.append(article_data)
                            articles.append(article_data)

                        # Stop if both targets are met
                        if len(company_articles) >= target_company and len(economic_articles) >= target_economic:
                            break

                    except Exception as e:
                        errors.append(f"Error processing article {entry.link}: {str(e)}")
                        continue

                if all_feeds_exhausted or (len(company_articles) >= target_company and len(economic_articles) >= target_economic):
                    break

            # Log if ratio wasn't perfectly met or fewer articles were fetched
            actual_company_ratio = len(company_articles) / len(articles) if articles else 0
            if abs(actual_company_ratio - company_ratio) > 0.1:  # Allow 10% deviation
                errors.append(
                    f"Desired company ratio {company_ratio:.2%} not met. "
                    f"Actual: {actual_company_ratio:.2%} ({len(company_articles)} company, {len(economic_articles)} economic). "
                    f"Consider using RSS feeds with more {'economic' if len(economic_articles) < target_economic else 'company'} content."
                )
            if len(articles) < max_articles:
                errors.append(
                    f"Fetched only {len(articles)} of {max_articles} articles due to insufficient "
                    f"{'economic' if len(economic_articles) < target_economic else 'company'} articles in provided feeds."
                )

            return {
                "success": True,
                "articles": articles,
                "errors": errors
            }

        except Exception as e:
            return {"success": False, "error": str(e), "articles": [], "errors": [str(e)]}

    async def fetch_news_from_websites(self, websites: List[Dict[str, str]], max_articles: int = 10) -> Dict[str, Any]:
        """Fetch news from websites using direct implementation"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from newspaper import Article

            articles = []

            for website in websites:
                try:
                    resp = requests.get(website["url"], timeout=10)
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    links = list({a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('http')})

                    for link in links[:max_articles]:
                        try:
                            art = Article(link)
                            art.download()
                            art.parse()

                            articles.append({
                                'source': website["name"],
                                'url': link,
                                'title': art.title,
                                'content': art.text,
                                'published_date': datetime.now().isoformat()
                            })
                        except Exception as e:
                            print(f"Error processing article {link}: {e}")
                            continue
                except Exception as e:
                    print(f"Error processing website {website['name']}: {e}")
                    continue

            return {"success": True, "articles": articles}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def classify_article(self, title: str, content: str) -> Dict[str, Any]:
        """Classify article using OpenAI"""
        try:
            prompt = f"""
            Classify the following news article as either 'economic' or 'company'.

            Title: {title}
            Content: {content[:1000]}

            Respond with only one word: 'economic' or 'company'.
            """

            result = call_openai(prompt).strip().lower()

            return {
                "success": True,
                "analysis": {
                    "classification": result,
                    "confidence": "high" if result in ["economic", "company"] else "low",
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def summarize_article(self, title: str, content: str, max_length: int = 100) -> Dict[str, Any]:
        """Summarize article using OpenAI"""
        try:
            prompt = f"""
            Summarize the following news article in {max_length} words or less.

            Title: {title}
            Content: {content[:2000]}

            Provide a clear, concise summary that captures the main points.
            """

            summary = call_openai(prompt).strip()

            return {
                "success": True,
                "summary": {
                    "summary": summary,
                    "word_count": len(summary.split()),
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def analyze_sentiment(self, title: str, content: str, summary: str = "") -> Dict[str, Any]:
        """Analyze sentiment using OpenAI"""
        try:
            text_to_analyze = summary if summary else content[:1000]

            prompt = f"""
            Analyze the sentiment of the following news content as either 'positive', 'negative', or 'neutral'.
            Also provide a brief reason for your sentiment analysis.

            Title: {title}
            Content: {text_to_analyze}

            Respond in the format: <sentiment> | <reason>
            """

            result = call_openai(prompt).strip()

            if '|' in result:
                sentiment, reason = result.split('|', 1)
                sentiment = sentiment.strip().lower()
                reason = reason.strip()
            else:
                sentiment = "neutral"
                reason = "Could not determine sentiment"

            return {
                "success": True,
                "sentiment": {
                    "sentiment": sentiment,
                    "reason": reason,
                    "confidence": "high" if sentiment in ["positive", "negative", "neutral"] else "low",
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def extract_company_info(self, title: str, content: str) -> Dict[str, Any]:
        """Extract company information using OpenAI"""
        try:
            prompt = f"""
            If the following article is about a specific company, extract the company name and any relevant company information.
            If it's not about a specific company, respond with 'No specific company mentioned'.

            Title: {title}
            Content: {content[:1000]}

            Respond with the company name if found, or 'No specific company mentioned'.
            """

            result = call_openai(prompt).strip()

            return {
                "success": True,
                "company_info": {
                    "company": result if result != "No specific company mentioned" else None,
                    "has_company_info": result != "No specific company mentioned",
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def analyze_impact(self, title: str, content: str, classification: str) -> Dict[str, Any]:
        """Analyze impact using OpenAI"""
        try:
            prompt = f"""
            Analyze the potential impact of the following news article.

            Title: {title}
            Content: {content[:1500]}
            Classification: {classification}

            Provide analysis in the following format:
            - Market Impact: (high/medium/low)
            - Sector Impact: (which sectors might be affected)
            - Time Horizon: (short-term/long-term)
            - Key Factors: (list main factors that could drive impact)
            """

            impact_analysis = call_openai(prompt).strip()

            return {
                "success": True,
                "impact": {
                    "impact_analysis": impact_analysis,
                    "classification": classification,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def save_articles_to_excel(self, articles: List[Dict[str, Any]], filename: str = "news_analysis.xlsx") -> Dict[str, Any]:
        """Save articles to Excel"""
        try:
            import pandas as pd

            if not articles:
                return {"success": False, "error": "No articles provided to save"}

            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            filepath = os.path.join("data", filename)

            # Convert to DataFrame and save
            df = pd.DataFrame(articles)
            df.to_excel(filepath, index=False)

            return {"success": True, "message": f"Successfully saved {len(articles)} articles to {filepath}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close connections (no-op for simplified client)"""
        pass

# Global MCP client instance
mcp_client = MCPClient()

# Convenience functions for AutoGen integration
async def mcp_fetch_news(feed_urls=None, websites=None, max_articles=10, company_ratio=0.7):
    """Convenience function to fetch news"""
    if websites:
        return await mcp_client.fetch_news_from_websites(websites, max_articles)
    elif feed_urls:
        return await mcp_client.fetch_news_from_rss(feed_urls, max_articles, company_ratio)
    else:
        return {"success": False, "error": "No RSS feeds or websites provided", "articles": [], "errors": []}

async def mcp_analyze_article(title, content):
    """Convenience function to analyze article"""
    results = {}

    # Classify article
    classification_result = await mcp_client.classify_article(title, content)
    if classification_result["success"]:
        results["classification"] = classification_result["analysis"]["classification"]

    # Summarize article
    summary_result = await mcp_client.summarize_article(title, content)
    if summary_result["success"]:
        results["summary"] = summary_result["summary"]["summary"]

    # Analyze sentiment
    sentiment_result = await mcp_client.analyze_sentiment(title, content, results.get("summary", ""))
    if sentiment_result["success"]:
        results["sentiment"] = sentiment_result["sentiment"]["sentiment"]
        results["sentiment_reason"] = sentiment_result["sentiment"]["reason"]

    # Extract company info if it's a company article
    if results.get("classification") == "company":
        company_result = await mcp_client.extract_company_info(title, content)
        if company_result["success"]:
            results["company"] = company_result["company_info"]["company"]

    # Analyze impact
    impact_result = await mcp_client.analyze_impact(title, content, results.get("classification", ""))
    if impact_result["success"]:
        results["impact_analysis"] = impact_result["impact"]["impact_analysis"]

    return results