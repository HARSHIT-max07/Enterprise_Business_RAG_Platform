import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from streamlit_option_menu import option_menu
from query_analytics import (
    get_query_history,
    get_daily_query_counts,
    get_quality_distribution,
    get_total_chunks,
    get_top_questions,
    get_daily_activity
)
import time
import os

from ingest_utils import ingest_pdf

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Enterprise Business RAG Platformm",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    

    st.title("🧠 Business RAG")

    st.markdown("---")
    
    st.subheader("🧭 Workspace")

    page = option_menu(
        menu_title=None,
        options=["Chat", "Analytics", "Knowledge Base"],
        icons=["chat-dots", "bar-chart", "database"],
        default_index=0,
    )

    st.subheader("⚙️ SYSTEM INFO")

    st.write("🤖 Model")
    st.write("Llama 3.2 (3B)")

    st.write("🗄️ Vector DB")
    st.write("Qdrant")

    st.write("🔎 Retriever")
    st.write("LangChain")

    st.write("⚡ Backend")
    st.write("FastAPI")

    st.write("🎈 Frontend")
    st.write("Streamlit")

    st.markdown("---")

    st.subheader("📊 KNOWLEDGE BASE")
    

    pdf_folder = "data/raw/business_docs"

    pdf_files = [
        f for f in os.listdir(pdf_folder)
        if f.endswith(".pdf")
    ]

    st.write(f"📄 Documents Indexed: {len(pdf_files)}")
    
    # =====================================
    # KNOWLEDGE BASE STATS
    # =====================================

    st.markdown("### 📊 Knowledge Base Stats")

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host="localhost",
            port=6333
        )

        collection_info = client.get_collection(
            "business_docs"
        )
        

        st.write(f"📚 Chunks: {collection_info.points_count}")
        st.caption("Collection: business_docs")

    except Exception as e:
        st.error(str(e))

    # =====================================

    with st.expander(f"📚 Indexed Documents ({len(pdf_files)})"):

        for pdf in pdf_files:

            col1, col2 = st.columns([4,1])

            with col1:
                st.write(pdf)

            with col2:
                if st.button(
                    "🗑",
                    key=f"delete_{pdf}"
                ):

                    pdf_path = os.path.join(
                        "data/raw/business_docs",
                        pdf
                    )

                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

                    st.success(f"{pdf} deleted!")

                    st.rerun()

    st.markdown("---")

    st.subheader("📄 Upload New PDF")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        save_path = os.path.join(
            "data/raw",
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(
            f"Saved: {uploaded_file.name}"
        )

        if st.button("🚀 Ingest Document"):

            with st.spinner(
                "Generating embeddings..."
            ):

                chunk_count = ingest_pdf(save_path)

            st.success(
                f"Successfully added {chunk_count} chunks"
            )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# ==========================================
# MAIN TITLE
# ==========================================
if page == "Chat":

    st.title("🧠 Enterprise Business RAG Platform")

    # ==========================================
    # WELCOME CARD
    # ==========================================

    if len(st.session_state.messages) == 0:

        st.markdown(
            """
            <div style="
            padding:12px 18px;
            border:1px solid #3b82f6;
            border-radius:12px;
            background-color:#0f172a;
            margin-bottom:25px;">

            <h2 style="
            color:#60a5fa;
            margin-top:0px;
            margin-bottom:6px;
            font-size:34px;
            font-weight:700;
            ">
            👋 Welcome to Enterprise Business RAG Platform
            </h2>

            <p style="
            font-size:14px;
            margin-top:0px;
            margin-bottom:6px;
            ">
            Ask business questions grounded in real company annual reports using AI-powered semantic search.
            </p>

            <p style="
            font-size:13px;
            opacity:0.85;
            margin-top:8px;
            margin-bottom:0px;
            ">
            🚀 Powered by:
            <b>Llama 3.2(3B)</b> |
            <b>LangChain</b> |
            <b>Qdrant</b> |
            <b>FastAPI</b> |
            <b>Streamlit</b>
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ==========================================
    # CHAT HISTORY
    # ==========================================

    for idx, message in enumerate(st.session_state.messages):

        with st.chat_message(message["role"]):
            
            if message["role"] == "assistant":

                query_type = message.get(
                    "query_type",
                    "General"
                )
                
                company = message.get(
                    "company",
                    ""
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    if message.get("is_comparison", False):

                        company_display = " | ".join(
                            [x.upper() for x in message.get("comparison_terms", [])]
                        )

                        st.metric(
                            "🏢 Companies",
                            company_display
                        )

                    else:

                        st.metric(
                            "🏢 Company",
                            str(company or "").upper()
                        )

                with col2:
                    st.metric(
                        "📊 Query Type",
                        query_type.replace("_", " ").title()
                    )

                with col3:
                    st.metric(
                        "🎯 Confidence",
                        f"{message.get('confidence', 0)}%"
                    )

                with col4:
                    st.metric(
                        "📚 Sources",
                        len(message.get("sources", []))
                    )

            st.markdown(message["content"])

            if "response_time" in message:
                if message["role"] == "assistant":

                    st.caption(
                        f"⏱ Response Time: {message['response_time']} sec | "
                        f"🔍 Retrieved Chunks: {len(message.get('sources', []))} |"
                        f"🎯 Confidence: {message.get('confidence', 0)}% "
                    )

            if message.get("sources"):

                with st.expander(
                    f"Sources ({len(message['sources'])})"
                ):

                    for source in message["sources"]:

                        source_name = source["source"]

                        if source_name.startswith("http"):

                            domain = (
                                source_name
                                .replace("https://", "")
                                .replace("http://", "")
                                .split("/")[0]
                            )

                            st.markdown(
                                f"""
                                <div style="
                                    padding:10px;
                                    border:1px solid #444;
                                    border-radius:8px;
                                    margin-bottom:8px;
                                ">
                                    🌐 <b>{domain}</b><br>
                                    Web Document
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            filename = source_name.split("/")[-1]

                            st.markdown(
                                f"""
                                <div style="
                                    padding:10px;
                                    border:1px solid #444;
                                    border-radius:8px;
                                    margin-bottom:8px;
                                ">
                                    📄 <b>{filename}</b><br>
                                    Page: {source['page']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                                    

    # ==========================================
    # QUESTION BOX (LARGER)
    # ==========================================

    question = st.text_area(
        "Ask an executive business question:",
        height=150,
        placeholder="Type your business question here..."
        
    )
    
    submit = st.button("🚀 Ask Question")
    
    if not question:
        st.markdown("""
    💡 **Try asking:**

    - Compare Meta and Alphabet
    - What are Nvidia's growth opportunities?
    - What risks does Tesla face?
    - Summarize Amazon's business segments
    """)


    # ==========================================
    # QUESTION HANDLING
    # ==========================================

    if submit and question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            start_time = time.time()

            try:

                with st.spinner(
                    "Searching knowledge base..."
                ):

                    response = requests.post(
                        "http://localhost:8000/chat",
                        json={
                            "question": question,
                            
                        }
                    )

                    result = response.json()

                    elapsed = round(
                        time.time() - start_time,
                        2
                    )
                    

                    query_type = result.get("query_type", "General")

                    company = result.get("company", "")
                    

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        if result.get("is_comparison", False):

                            company_display = " | ".join(
                                [x.upper() for x in result.get("comparison_terms", [])]
                            )

                            st.metric(
                                "📊 Companies",
                                company_display
                            )

                        else:

                            st.metric(
                                "🏢 Company",
                                company.upper()
                            )

                    with col2:
                        st.metric(
                            "📊 Query Type",
                            query_type.replace(
                                "_",
                                " "
                            ).title()
                        )

                    with col3:
                        st.metric(
                            "🎯 Confidence",
                            f"{result.get('confidence', 0)}%"
                        )

                    with col4:
                        st.metric(
                            "📚 Sources",
                            len(result["sources"])
                        )

                    st.markdown(result["answer"])

                    quality = result.get("quality", "Unknown")

                    if quality == "High Quality":
                        badge = "🟢 High Quality"

                    elif quality == "Medium Quality":
                        badge = "🟡 Medium Quality"

                    else:
                        badge = "🔴 Low Quality"

                    st.caption(
                        f"⏱ Response Time: {elapsed} sec | "
                        f"🔍 Retrieved Chunks: {len(result['sources'])} |"
                        f"🎯 Confidence: {result.get('confidence', 0)}% | "
                        f"{badge}"
                    )

                    with st.expander(
                        f"Sources ({len(result['sources'])})"
                    ):

                        for source in result["sources"]:

                            source_name = source["source"]

                            if source_name.startswith("http"):

                                domain = (
                                    source_name
                                    .replace("https://", "")
                                    .replace("http://", "")
                                    .split("/")[0]
                                )

                                st.markdown(
                                    f"""
                                    <div style="
                                        padding:10px;
                                        border:1px solid #444;
                                        border-radius:8px;
                                        margin-bottom:8px;
                                    ">
                                        🌐 <b>{domain}</b><br>
                                        Web Document
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            else:

                                filename = source_name.split("/")[-1]

                                st.markdown(
                                    f"""
                                    <div style="
                                        padding:10px;
                                        border:1px solid #444;
                                        border-radius:8px;
                                        margin-bottom:8px;
                                    ">
                                        📄 <b>{filename}</b><br>
                                        Page: {source['page']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

            except Exception as e:
                st.error(f"REAL ERROR: {str(e)}")

        st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "response_time": elapsed,
            "confidence": result.get("confidence", 0),
            "quality": result.get("quality", "Unknown"),
            "query_type": result.get("query_type", "General"),
            "company": result.get("company", ""),

            "is_comparison": result.get("is_comparison", False),
            "comparison_terms": result.get("comparison_terms", [])
        }
        )
        
        st.rerun()

    # ==========================================
    # FOOTER
    # ==========================================

    st.markdown("---")

    st.caption(
        " 🧠 Enterprise Business RAG Platform | "
        "Built with Llama 3.2(3B) + LangChain + Qdrant + FastAPI + Streamlit"
    )
# =====================================
# ANALYTICS DASHBOARD
# =====================================

if page == "Analytics":

    st.divider()

    st.subheader("📊 Query Analytics")

    df = get_query_history()
    
    top_questions = get_top_questions()

    if not df.empty:

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        with col1:
            st.metric(
                "Total Queries",
                len(df)
            )

        with col2:
            st.metric(
                "Avg Response Time",
                round(df["response_time"].mean(), 2)
            )

        with col3:
            st.metric(
                "Avg Confidence",
                round(df["confidence"].mean(), 2)
            )
            
        with col4:
            st.metric(
                "High Quality",
                len(df[df["quality"] == "High Quality"])
            )

        with col5:
            st.metric(
                "Medium Quality",
                len(df[df["quality"] == "Medium Quality"])
            )

        with col6:
            st.metric(
                "Low Quality",
                len(df[df["quality"] == "Low Quality"])
            )    

        st.subheader("Recent Queries")

        st.dataframe(
            df.sort_values(
                "id",
                ascending=False
            ),
            hide_index=True,
            use_container_width=True
        )
        st.subheader("🔥 Most Asked Questions")

        if not top_questions.empty:

            top_questions.insert(
                0,
                "Rank",
                range(1, len(top_questions) + 1)
            )

            st.dataframe(
                top_questions,
                hide_index=True,
                use_container_width=True
            )
            
        st.subheader("📈 Query Trends")

        trend_df = get_daily_query_counts()

        if not trend_df.empty:

            fig = px.line(
                trend_df,
                x=trend_df["date"].astype(str),
                y="total_queries",
                markers=True,
                title="Queries Over Time"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
            
        st.subheader("📈 Confidence Trend")

        confidence_chart = px.line(
            df,
            x="timestamp",
            y="confidence",
            markers=True,
            title="Confidence Over Time"
        )

        st.plotly_chart(
            confidence_chart,
            use_container_width=True
        )    
        
        st.subheader("⚡ Response Time Trend")

        response_chart = px.line(
            df,
            x="timestamp",
            y="response_time",
            markers=True,
            title="Response Time Over Time"
        )

        st.plotly_chart(
            response_chart,
            use_container_width=True
        )
        
        st.subheader("📅 Daily Query Activity")

        activity_df = get_daily_activity()

        if not activity_df.empty:

            activity_chart = px.bar(
                activity_df,
                x="date",
                y="queries",
                title="Daily Query Activity"
            )

            st.plotly_chart(
                activity_chart,
                use_container_width=True
            )
            
        st.subheader("🎯 Answer Quality Distribution")

        quality_df = get_quality_distribution()

        if not quality_df.empty:

            fig = px.pie(
                quality_df,
                names="quality",
                values="count",
                title="Quality Breakdown"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )   

if page == "Knowledge Base":

    st.title("📚 Knowledge Base")

    st.subheader("System Statistics")
    
    pdf_folder = "data/raw/business_docs"


    documents = sorted([
        file
        for file in os.listdir(pdf_folder)
        if file.endswith(".pdf")
    ])
    
    total_chunks = get_total_chunks()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Companies Indexed",
            len(documents)
        )

    with col2:
        st.metric(
            "Documents Indexed",
            len(documents)
        )

    with col3:
        st.metric(
            "Collection",
            "business_docs"
        )

    with col4:
        st.metric(
            "Total Chunks",
            total_chunks
        )   
    st.divider()

    st.subheader("📋 Indexed Companies")

    company_data = []

    for doc in documents:

        company_name = (
            doc.replace(".pdf", "")
            .replace("_Annual_Report_2025", "")
        )

        company_data.append(
            {
                "Company": company_name,
                "Document": doc
            }
        )

    df = pd.DataFrame(company_data)

    df.index = df.index + 1

    st.dataframe(
        df,
        use_container_width=True
    )
    st.divider()    
    
    st.subheader("📊 Business Knowledge Base")

    st.info(
        f"""
        Companies Indexed: {len(documents)}

        Collection: business_docs

        Embedding Model: nomic-embed-text

        Vector Database: Qdrant
        """
    )