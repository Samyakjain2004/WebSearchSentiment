from autogen import AssistantAgent
import os
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI for new API 
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

# Check if we have Azure OpenAI or regular OpenAI credentials
if os.getenv("AZURE_OPENAI_API_KEY"):
    llm_config = {
        "config_list": [
            {
                "model": os.getenv("AZURE_OPENAI_MODEL_NAME"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_type": "azure",
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            },
        ],
        "temperature": 0.3,
    }
else:
    # Fallback to regular OpenAI
    llm_config = {
        "config_list": [
            {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        ],
        "temperature": 0.3,
    }

classifier_agent = AssistantAgent(name="Classifier", llm_config=llm_config)
summarizer_agent = AssistantAgent(name="Summarizer", llm_config=llm_config)
sentiment_agent = AssistantAgent(name="Sentiment", llm_config=llm_config)
reason_agent = AssistantAgent(name="Reasoner", llm_config=llm_config)

# Agent role functions for classification, summarization, sentiment, and company extraction
# Integrate with AutoGen/MCP for LLM-based analysis

def call_azure_openai(prompt):
    """Call Azure OpenAI API with proper error handling"""
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
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512
            )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error: Could not process request"

def classify_article(article):
    prompt = f"""
    Classify the following news article as either 'economy' or 'company'.\n
    Title: {article.get('title', 'No title')}\n
    Content: {article.get('content', '')[:500]}\n
    Respond with only one word: 'economy' or 'company'.
    """
    result = call_azure_openai(prompt).strip().lower()
    if 'company' in result:
        return 'company'
    return 'economy'

def summarize_article(article):
    prompt = f"""
    Summarize the following news article in 2-3 sentences.\n
    Title: {article.get('title', 'No title')}\n
    Content: {article.get('content', '')[:1000]}
    """
    return call_azure_openai(prompt).strip()

def analyze_sentiment(article, summary):
    prompt = f"""
    Analyze the sentiment of the following news summary as either '+ve' (positive) or '-ve' (negative) for the economy or company. Also, provide a brief reason for your sentiment.\n
    Summary: {summary}\n
    Respond in the format: <sentiment> | <reason>
    """
    result = call_azure_openai(prompt).strip()
    if '|' in result:
        sentiment, reason = result.split('|', 1)
        return sentiment.strip(), reason.strip()
    return '-ve', 'Could not determine sentiment.'

def extract_company_info(article):
    prompt = f"""
    If the following article is about a specific company, extract the company name. Otherwise, respond with '-'.\n
    Title: {article.get('title', 'No title')}\n
    Content: {article.get('content', '')[:500]}
    """
    result = call_azure_openai(prompt).strip()
    return result if result and result != '-' else '-'