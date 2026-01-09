# WebSearchSentiment

**WebSearchSentiment** is a Python-based system that performs automated web searches (via RSS feeds and potentially search APIs), collects content, and analyzes sentiment using language models / AI agents.  
It appears to use a multi-agent and tool-calling architecture.

## ✨ Features (based on current structure)

- RSS feed monitoring & content aggregation
- AI-powered agent system for processing web content
- Sentiment analysis on collected articles/news/posts
- Modular architecture with separate agents, utilities and server components
- Streamlit / web interface (via `app.py`)
- Extensible agent controller system

## 🛠️ Tech Stack

- **Python** ≥ 3.10 (recommended)
- **Dependencies** — see [`requirements.txt`](./requirements.txt)
- **Frontend** — Streamlit (`app.py`)
- **Architecture** — Agent-based with MCP (Model Context Protocol) servers and client

## Quick Start

### Clone the repository

```bash
git clone https://github.com/Samyakjain2004/WebSearchSentiment.git
cd WebSearchSentiment
```
## Setup Instructions

### 1. Environment Setup

**Option A: Using Virtual Environment (Recommended)**
```bash
# Install python3-venv (if not available)
sudo apt update
sudo apt install python3.12-venv

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

**Option B: System-wide Installation**
```bash
# Install dependencies directly
python3 -m pip install -r requirements.txt
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables Configuration

Edit the `.env` file with your actual API credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_actual_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-service-name.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=your_gpt4_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-01
AUTOGEN_USE_DOCKER=False
OPENAI_API_KEY=your_actual_openai_api_key
NEWS_API_KEY=your_actual_news_api_key
```

**Getting API Keys:**
- **Azure OpenAI**: Sign up at https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/

### 4. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`
Some commonly available RSS URLs are
```bash
[streamlit run app.py](https://finance.yahoo.com/news/rssindex
https://news.google.com/rss/search?q=company+OR+earnings&hl=en-IN&gl=IN&ceid=IN:en
https://www.moneycontrol.com/rss/latestnews.xml
https://www.cnbc.com/id/100003114/device/rss/rss.html
https://timesofindia.indiatimes.com/rssfeedstopstories.cms
https://www.thehindu.com/news/national/feeder/default.rss
https://feeds.feedburner.com/ndtvnews-top-stories
https://rss.nytimes.com/services/xml/rss/nyt/World.xml
https://www.indiatoday.in/rss/home
```

## Project Structure

```
WebSearchSentiment/
├── agents/               # Different agent implementations / roles
├── data/                 # Cached articles, results, sentiment outputs
├── mcp_servers/          # Model context protocol / chain servers
├── utils/                # Helper functions, text processing, sentiment utils
│
├── app.py                # Main Streamlit application
├── agent_controller.py   # Orchestrates / controls the agents
├── mcp_client.py         # Client to communicate with MCP servers
├── requirements.txt      # All Python dependencies
├── rss_urls              # List of RSS feeds to monitor (txt or json)
└── README.md
```
## ⚙️ How It Works

- Collects latest items from RSS feeds listed in rss_urls
- Processes articles using one or multiple AI agents
- Performs sentiment analysis (positive/neutral/negative) + possibly intensity scoring
- Stores results (in data/ folder)
- Shows results/visualizations in Streamlit dashboard

### 📜 License

This project is licensed under the MIT License.

### 👨‍💻 Author

Samyak Jain
🔗 LinkedIn - https://www.linkedin.com/in/samyak-jain-470b7b255

🔗 GitHub - https://github.com/Samyakjain2004

## Next Steps
- Add real-time web search (SerpAPI, Tavily, SearxNG, etc.)
- Support more LLM providers (Groq, Anthropic, local Ollama, DeepSeek…)
- Add visualization of sentiment trends over time
- Export reports (PDF/CSV)
- Schedule automatic analysis (cron / APScheduler)
- Add keyword filtering & topic clustering
- Better error handling & logging
