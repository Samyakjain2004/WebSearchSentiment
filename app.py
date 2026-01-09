import streamlit as st
from agent_controller import run_agents_and_generate_output
import pandas as pd

st.set_page_config(page_title="AI NEWS IMPACT ANALYZER", layout="wide")

# --- Custom Header ---
st.markdown("""
    <div style='text-align: center; margin-top: 0px;'>
        <h2 style='font-size: 35px; font-family: Courier New, monospace;'>
            <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" width="60" height="50">
            <span style='background: linear-gradient(45deg, #ed4965, #c05aaf); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                AI NEWS IMPACT ANALYZER
            </span>
        </h2>
    </div>
    """, unsafe_allow_html=True)

st.write("### Configure Your Analysis")

# --- Input Layout in Columns ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    num_urls = st.number_input(
        "No. of RSS URLs",
        min_value=0,
        max_value=10,
        value=1,
        help="Enter 0 to use fallback method."
    )

with col2:
    max_articles = st.number_input(
        "Total Articles",
        min_value=1,
        max_value=100,
        value=10,
        help="Total articles to fetch from all feeds."
    )

with col3:
    company_ratio = st.slider(
        "Company Articles (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Remainder will be economic."
    )

# --- RSS URL Inputs in Expander ---
url_list = []
if num_urls > 0:
    with st.expander("Enter RSS Feed URLs", expanded=False):
        for i in range(num_urls):
            url = st.text_input(
                f"RSS Feed URL {i + 1}",
                placeholder="https://example.com/rss",
                key=f"rss_url_{i}",
            )
            if url.strip():
                url_list.append(url.strip())

# --- Action Button ---
if st.button("Fetch and Analyze News", use_container_width=True):
    st.info("⏳ Running agents and fetching news...")

    output = run_agents_and_generate_output(url_list, max_articles=int(max_articles), company_ratio=company_ratio/100)

    if not output or "excel" not in output:
        st.error("❌ Failed to fetch or analyze news. Check logs.")
        if output and "errors" in output:
            with st.expander("View Fetch Errors", expanded=False):
                for error in output["errors"]:
                    st.error(error)
    else:
        st.success("✅ Analysis complete!")

        if output.get("errors"):
            with st.expander("View Fetch Warnings", expanded=False):
                for error in output["errors"]:
                    st.warning(error)

        # Display tables in tabs
        tab1, tab2 = st.tabs(["Economy/Company News", "Company Level News"])

        with tab1:
            df1 = pd.read_excel(output["excel"], sheet_name="Economy_Company")
            st.dataframe(df1, use_container_width=True)
            st.download_button(
                "⬇️ Download Table",
                output["excel"],
                file_name="news_impact_analysis.xlsx",
            )

        with tab2:
            df2 = pd.read_excel(output["excel"], sheet_name="Company_Level")
            st.dataframe(df2, use_container_width=True)

else:
    st.info("💡 Configure your inputs and click **Fetch and Analyze News**.")
