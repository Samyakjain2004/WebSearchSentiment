from autogen import UserProxyAgent, GroupChat, GroupChatManager
from agents.agentsroles import classifier_agent, summarizer_agent, sentiment_agent, reason_agent
from utils.scraper import fetch_articles_from_source
from newspaper import Article
from dotenv import load_dotenv
import pandas as pd
import datetime
import os
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from mcp_client import mcp_client, mcp_fetch_news, mcp_analyze_article

load_dotenv()

# Set AutoGen to not use Docker
os.environ["AUTOGEN_USE_DOCKER"] = "False"

user_proxy = UserProxyAgent(name="User", human_input_mode="NEVER")

groupchat = GroupChat(
    agents=[user_proxy, classifier_agent, summarizer_agent, sentiment_agent, reason_agent],
    messages=[],
    max_round=5,
)
chat_manager = GroupChatManager(groupchat=groupchat)

async def run_agents_with_mcp(rss_urls=None, max_articles=5, company_ratio=0.7):
    """Main function to fetch articles and generate analysis output using MCP with ratio"""
    try:
        print("Initializing MCP servers...")
        await mcp_client.initialize_servers()

        print("Fetching news using MCP...")
        # Fetch news using MCP with user-provided RSS URLs, max_articles, and company_ratio
        news_result = await mcp_fetch_news(feed_urls=rss_urls, max_articles=max_articles, company_ratio=company_ratio)

        errors = []
        if not news_result["success"]:
            errors.append(f"Error fetching news: {news_result['error']}")
            return {"errors": errors}

        articles = news_result["articles"]
        errors.extend(news_result.get("errors", []))  # Collect any article-specific errors
        print(f"Fetched {len(articles)} articles")

        if not articles:
            print("No articles fetched. Using fallback method...")
            articles = fetch_articles_fallback(max_articles)
            if not articles:
                errors.append("Fallback method failed to fetch articles")
                return {"errors": errors}

        if articles:
            print("Analyzing articles using MCP...")
            analyzed_articles = []

            for i, article in enumerate(articles):
                print(f"Analyzing article {i+1}/{len(articles)}: {article.get('title', 'No title')[:50]}...")

                # Analyze article using MCP
                analysis = await mcp_analyze_article(
                    article.get('title', ''),
                    article.get('content', '')
                )

                # Combine article data with analysis
                analyzed_article = {
                    'url': article.get('url', ''),
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'published_date': article.get('published_date', ''),
                    'classification': analysis.get('classification', 'unknown'),
                    'summary': analysis.get('summary', ''),
                    'sentiment': analysis.get('sentiment', 'neutral'),
                    'sentiment_reason': analysis.get('sentiment_reason', ''),
                    'company': analysis.get('company', ''),
                    'impact_analysis': analysis.get('impact_analysis', ''),
                    'analysis_date': datetime.datetime.now().isoformat()
                }

                analyzed_articles.append(analyzed_article)

            # Separate articles by type
            economy_articles = [a for a in analyzed_articles if a['classification'] == 'economic']
            company_articles = [a for a in analyzed_articles if a['classification'] == 'company']

            # Create DataFrames
            economy_company_table = pd.DataFrame(analyzed_articles)
            company_level_table = pd.DataFrame(company_articles)

            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)

            output_path = os.path.join("data", "output.xlsx")
            with pd.ExcelWriter(output_path) as writer:
                economy_company_table.to_excel(writer, sheet_name="Economy_Company", index=False)
                company_level_table.to_excel(writer, sheet_name="Company_Level", index=False)

            print(f"Analysis complete. Output saved to: {output_path}")
            print(f"Total articles analyzed: {len(analyzed_articles)}")
            print(f"Economic articles: {len(economy_articles)}")
            print(f"Company articles: {len(company_articles)}")

            return {"excel": output_path, "errors": errors}
        else:
            errors.append("No articles could be fetched")
            return {"errors": errors}
    except Exception as e:
        print(f"Error in run_agents_with_mcp: {e}")
        return {"errors": [f"General error: {str(e)}"]}

def run_agents_and_generate_output(rss_urls=None, max_articles=5, company_ratio=0.7):
    """Synchronous wrapper for the async MCP function"""
    try:
        return asyncio.run(run_agents_with_mcp(rss_urls, max_articles, company_ratio))
    except Exception as e:
        print(f"Error in run_agents_and_generate_output: {e}")
        return {"errors": [f"General error: {str(e)}"]}

def fetch_articles_fallback(max_articles=5):
    """Fallback method to fetch articles if the main method fails"""
    # Fallback to a default website since RSS feeds are now user-provided
    articles = []
    try:
        # Use a generic news website as fallback
        articles = fetch_articles_from_source("fallback", "https://www.reuters.com/", max_articles=max_articles)
    except Exception as e:
        print(f"Error in fallback method: {e}")
    return articles

# Initialize scheduler for periodic updates
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: run_agents_and_generate_output(), 'interval', hours=0.1)
scheduler.start()

# Cleanup function for MCP client
async def cleanup_mcp():
    """Cleanup MCP client connections"""
    await mcp_client.close()

def cleanup_mcp_sync():
    """Synchronous wrapper for MCP cleanup"""
    try:
        asyncio.run(cleanup_mcp())
    except Exception as e:
        print(f"Error during MCP cleanup: {e}")

# Register cleanup on exit
import atexit
atexit.register(cleanup_mcp_sync)