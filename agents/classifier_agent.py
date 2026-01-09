import pandas as pd
from agents.agentsroles import classify_article, summarize_article, analyze_sentiment, extract_company_info

def classify_and_analyze_news(articles):
    rows = []
    company_rows = []
    for article in articles:
        article_type = classify_article(article)
        summary = summarize_article(article)
        sentiment, reason = analyze_sentiment(article, summary)
        company = extract_company_info(article) if article_type == "company" else "-"
        rows.append({
            "url": article["url"],
            "title": article["title"],
            "summary": summary,
            "type": article_type,
            "sentiment": sentiment,
            "reason": reason
        })
        if article_type == "company":
            company_rows.append({
                "company": company,
                "url": article["url"],
                "summary": summary,
                "sentiment": sentiment,
                "reason": reason
            })
    df1 = pd.DataFrame(rows)
    df2 = pd.DataFrame(company_rows)
    return df1, df2 