
""" MCP Server for AI Analysis
Provides LLM-based tools for news analysis
"""

import asyncio
import json
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
import openai
import os
from dotenv import load_dotenv
from datetime import datetime
from mcp import ClientSession, StdioServerParameters

# Load environment variables
load_dotenv()

# Initialize MCP server
server = Server("ai-analysis-server")


# Configure OpenAI
if os.getenv("AZURE_OPENAI_API_KEY"):
    openai.api_type = "azure"
    openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
    openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
    openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_MODEL_NAME')
else:
    openai.api_key = os.getenv('OPENAI_API_KEY')

    

def call_openai(prompt: str, model: str = None) -> str:
    """Call OpenAI API with proper error handling"""
    try:
        if os.getenv("AZURE_OPENAI_API_KEY"):
            # Azure OpenAI
            response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512
            )
        else:
            # Regular OpenAI
            model = model or "gpt-3.5-turbo"
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512
            )
        
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available AI analysis tools"""
    return ListToolsResult(
        tools=[
            Tool(
                name="classify_article",
                description="Classify a news article as 'economic' or 'company' using AI",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the article"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the article"
                        }
                    },
                    "required": ["title", "content"]
                }
            ),
            Tool(
                name="summarize_article",
                description="Generate a concise summary of a news article using AI",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the article"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the article"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum length of summary in words",
                            "default": 100
                        }
                    },
                    "required": ["title", "content"]
                }
            ),
            Tool(
                name="analyze_sentiment",
                description="Analyze the sentiment of a news article using AI",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the article"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the article"
                        },
                        "summary": {
                            "type": "string",
                            "description": "Summary of the article (optional)"
                        }
                    },
                    "required": ["title", "content"]
                }
            ),
            Tool(
                name="extract_company_info",
                description="Extract company information from a news article using AI",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the article"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the article"
                        }
                    },
                    "required": ["title", "content"]
                }
            ),

            Tool(
                name="analyze_impact",
                description="Analyze the potential impact of a news article using AI",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the article"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the article"
                        },
                        "classification": {
                            "type": "string",
                            "description": "Classification of the article (economic/company)"
                        }
                    },
                    "required": ["title", "content"]
                }
            )
        ]
    )


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> CallToolResult:
    """Handle tool calls for AI analysis"""
    
    if name == "classify_article":
        return await classify_article(arguments or {})
    elif name == "summarize_article":
        return await summarize_article(arguments or {})
    elif name == "analyze_sentiment":
        return await analyze_sentiment(arguments or {})
    elif name == "extract_company_info":
        return await extract_company_info(arguments or {})
    elif name == "analyze_impact":
        return await analyze_impact(arguments or {})
    else:
        raise ValueError(f"Unknown tool: {name}")


async def classify_article(args: dict) -> CallToolResult:
    """Classify article as economic or company"""
    title = args.get("title", "")
    content = args.get("content", "")
    
    prompt = f"""
    Classify the following news article as either 'economic' or 'company'.
    
    Title: {title}
    Content: {content[:1000]}
    
    Respond with only one word: 'economic' or 'company'.
    """
    
    result = call_openai(prompt).strip().lower()
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps({
                    "classification": result,
                    "confidence": "high" if result in ["economic", "company"] else "low",
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
            )
        ]
    )


async def summarize_article(args: dict) -> CallToolResult:
    """Summarize article content"""
    title = args.get("title", "")
    content = args.get("content", "")
    max_length = args.get("max_length", 100)
    
    prompt = f"""
    Summarize the following news article in {max_length} words or less.
    
    Title: {title}
    Content: {content[:2000]}
    
    Provide a clear, concise summary that captures the main points.
    """
    
    summary = call_openai(prompt).strip()
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps({
                    "summary": summary,
                    "word_count": len(summary.split()),
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
            )
        ]
    )

async def analyze_sentiment(args: dict) -> CallToolResult:
    """Analyze article sentiment"""
    title = args.get("title", "")
    content = args.get("content", "")
    summary = args.get("summary", "")
    
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
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps({
                    "sentiment": sentiment,
                    "reason": reason,
                    "confidence": "high" if sentiment in ["positive", "negative", "neutral"] else "low",
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
            )
        ]
    )

async def extract_company_info(args: dict) -> CallToolResult:
    """Extract company information from article"""
    title = args.get("title", "")
    content = args.get("content", "")
    
    prompt = f"""
    If the following article is about a specific company, extract the company name and any relevant company information.
    If it's not about a specific company, respond with 'No specific company mentioned'.
    
    Title: {title}
    Content: {content[:1000]}
    
    Respond with the company name if found, or 'No specific company mentioned'.
    """
    
    result = call_openai(prompt).strip()
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps({
                    "company": result if result != "No specific company mentioned" else None,
                    "has_company_info": result != "No specific company mentioned",
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
            )
        ]
    )

async def analyze_impact(args: dict) -> CallToolResult:
    """Analyze potential impact of the article"""
    title = args.get("title", "")
    content = args.get("content", "")
    classification = args.get("classification", "")
    
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
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps({
                    "impact_analysis": impact_analysis,
                    "classification": classification,
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
            )
        ]
    )

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ai-analysis-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 