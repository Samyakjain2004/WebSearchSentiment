"""
MCP Server for News Fetching Provides tools for fetching and processing news articles
"""

import asyncio
import json
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ( CallToolResult, ListToolsResult,Tool,TextContent)
import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from datetime import datetime
import pandas as pd
import os

# Initialize MCP server
server = Server("news-server")

@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available tools for news fetching and processing"""
    return ListToolsResult(
        tools=[
            Tool(
                name="fetch_news_from_rss",
                description="Fetch news articles from RSS feeds",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feed_urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of RSS feed URLs to fetch from"
                        },
                        "max_articles": {
                            "type": "integer",
                            "description": "Maximum number of articles to fetch per feed",
                            "default": 10
                        }
                    },
                    "required": ["feed_urls"]
                }
            ),
            Tool(
                name="fetch_news_from_websites",
                description="Fetch news articles from specific websites",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "websites": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "url": {"type": "string"}
                                },
                                "required": ["name", "url"]
                            },
                            "description": "List of websites to scrape"
                        },
                        "max_articles": {
                            "type": "integer",
                            "description": "Maximum number of articles to fetch per website",
                            "default": 10
                        }
                    },
                    "required": ["websites"]
                }
            ),
            Tool(
                name="analyze_article",
                description="Analyze a single article for classification, sentiment, and summary",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the article to analyze"
                        },
                        "title": {
                            "type": "string",
                            "description": "Title of the article (optional)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the article (optional)"
                        }
                    },
                    "required": ["url"]
                }
            ),
            Tool(
                name="save_articles_to_excel",
                description="Save analyzed articles to Excel file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "articles": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of analyzed articles"
                        },
                        "filename": {
                            "type": "string",
                            "description": "Name of the Excel file",
                            "default": "news_analysis.xlsx"
                        }
                    },
                    "required": ["articles"]
                }
            )
        ]
    )

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> CallToolResult:
    """Handle tool calls for news operations"""
    
    if name == "fetch_news_from_rss":
        return await fetch_news_from_rss(arguments or {})
    elif name == "fetch_news_from_websites":
        return await fetch_news_from_websites(arguments or {})
    elif name == "analyze_article":
        return await analyze_article(arguments or {})
    elif name == "save_articles_to_excel":
        return await save_articles_to_excel(arguments or {})
    else:
        raise ValueError(f"Unknown tool: {name}")

async def fetch_news_from_rss(args: dict) -> CallToolResult:
    """Fetch news from RSS feeds"""
    feed_urls = args.get("feed_urls", [])
    max_articles = args.get("max_articles", 10)
    
    articles = []
    
    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_articles]:
                try:
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    
                    articles.append({
                        'url': entry.link,
                        'title': article.title or entry.title,
                        'content': article.text,
                        'source': feed_url,
                        'published_date': entry.get('published', ''),
                        'summary': entry.get('summary', '')
                    })
                except Exception as e:
                    print(f"Error processing article {entry.link}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing feed {feed_url}: {e}")
            continue
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Successfully fetched {len(articles)} articles from RSS feeds"
            ),
            TextContent(
                type="text",
                text=json.dumps(articles, indent=2, default=str)
            )
        ]
    )

async def fetch_news_from_websites(args: dict) -> CallToolResult:
    """Fetch news from specific websites"""
    websites = args.get("websites", [])
    max_articles = args.get("max_articles", 10)
    
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
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Successfully fetched {len(articles)} articles from websites"
            ),
            TextContent(
                type="text",
                text=json.dumps(articles, indent=2, default=str)
            )
        ]
    )

async def analyze_article(args: dict) -> CallToolResult:
    """Analyze a single article"""
    url = args.get("url")
    title = args.get("title", "")
    content = args.get("content", "")
    
    if not content and url:
        try:
            article = Article(url)
            article.download()
            article.parse()
            title = article.title
            content = article.text
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error fetching article: {e}"
                    )
                ]
            )
    
    analysis = {
        'url': url,
        'title': title,
        'content_length': len(content),
        'word_count': len(content.split()),
        'has_economic_keywords': any(keyword in content.lower() for keyword in ['economy', 'inflation', 'gdp', 'fed', 'interest']),
        'has_company_keywords': any(keyword in content.lower() for keyword in ['company', 'earnings', 'stock', 'market', 'business']),
        'classification': 'economic' if any(keyword in content.lower() for keyword in ['economy', 'inflation', 'gdp', 'fed']) else 'company',
        'summary': content[:200] + "..." if len(content) > 200 else content,
        'analysis_date': datetime.now().isoformat()
    }
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(analysis, indent=2)
            )
        ]
    )

async def save_articles_to_excel(args: dict) -> CallToolResult:
    """Save articles to Excel file"""
    articles = args.get("articles", [])
    filename = args.get("filename", "news_analysis.xlsx")
    
    if not articles:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text="No articles provided to save"
                )
            ]
        )
    
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(articles)
        df.to_excel(filepath, index=False)
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Successfully saved {len(articles)} articles to {filepath}"
                )
            ]
        )
    except Exception as e:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Error saving articles: {e}"
                )
            ]
        )

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="news-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 