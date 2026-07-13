from app.hybrid_search import bm25_search
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from ollama import Client
from qdrant_client import QdrantClient

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest
)
import os
import re
import traceback
import time
import sqlite3
from datetime import datetime


# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="Enterprise RAG API"
)


# ==========================================
# OLLAMA CLIENT
# ==========================================

print("Connecting to Ollama...")

ollama_client = Client(
    host="http://localhost:11434"
)


# ==========================================
# QDRANT CLIENT
# ==========================================

print("Connecting to Qdrant...")

qdrant = QdrantClient(
    host="localhost",
    port=6333
)


# ==========================================
# PROMETHEUS METRICS
# ==========================================

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total RAG Requests"
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "RAG Request Latency"
)

PDF_QUERIES = Counter(
    "pdf_queries_total",
    "Total PDF Queries"
)

WEB_QUERIES = Counter(
    "web_queries_total",
    "Total Web Queries"
)

COMPARISON_QUERIES = Counter(
    "comparison_queries_total",
    "Total Comparison Queries"
)


# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):
    question: str


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Enterprise RAG Knowledge Platform Running"
    }


# ==========================================
# CHAT ENDPOINT
# ==========================================
def detect_company(question):
    question = question.lower()
    
    if "google" in question:
        return "alphabet"

    if "aws" in question:
        return "amazon"

    if "azure" in question:
        return "microsoft"

    if "facebook" in question:
        return "meta"
    if "hcl tech" in question:
        return "hcltech"

    if "hcl technologies" in question:
        return "hcltech"

    if "service now" in question:
        return "servicenow"

    if "qcom" in question:
        return "qualcomm"
    
    # Companies not present in KB
    if "tcs" in question:
        return "__UNKNOWN__"

    if "infosys" in question:
        return "__UNKNOWN__"

    companies = [
        "amazon",
        "microsoft",
        "alphabet",
        "google",
        "meta",
        "apple",
        "nvidia",
        "tesla",
        "netflix",
        "adobe",
        "oracle",
        "salesforce",
        "amd",
        "intel",

        "aws",
        "azure",
        "microsoft cloud",
        "google cloud",

        "qualcomm",
        "walmart",
        "visa",
        "paypal",
        "cognizant",
        "accenture",

        "hcltech",
        "hcl tech",
        "hcl technologies",

        "ibm",
        "cisco",

        "servicenow",
        "service now",

        "uber"
    ]

    for company in companies:
        if company in question:
            return company  

    return None

@app.post("/chat")
def chat_endpoint(request: ChatRequest):

    start_time = time.time()

    REQUEST_COUNT.inc()
       

    try:

        print("\n==============================")
        print("NEW REQUEST")
        print("==============================")

        print("Question:")
        print(request.question)

        # ----------------------------------
        # Generate Embedding
        # ----------------------------------

        print("\nGenerating embedding...")

        query_embedding = ollama_client.embeddings(
            model="nomic-embed-text",
            prompt=request.question
        )["embedding"]
        
        print("Embedding generated")

        # ----------------------------------
        # Comparison Detection
        # ----------------------------------

        question = request.question.lower()

        question_lower = question

        query_type = "general"

        if any(term in question_lower for term in [
            "cloud",
            "aws",
            "azure",
            "google cloud",
            "cloud services",
            "cloud platform",
            "cloud business",
            "cloud revenue"
        ]):
            query_type = "cloud"

        elif (
            re.search(r"\bai\b", question_lower)
            or "artificial intelligence" in question_lower
            or "generative ai" in question_lower
            or "machine learning" in question_lower
            or "copilot" in question_lower
            or "openai" in question_lower
            or "llm" in question_lower
        ):
            query_type = "ai"
            
        elif any(term in question_lower for term in [
            "business segment",
            "business segments",
            "segment",
            "segments"
            
        ]):
            query_type = "segment"    
            
        elif any(term in question_lower for term in [
            "investment",
            "investments",
            "investing",
            "capital allocation",
            "acquisition",
            "acquisitions"
        ]):
            query_type = "investment"    
            
        elif any(term in question_lower for term in [
            "strategy",
            "strategic",
            "vision",
            "roadmap",
            "long term"
        ]):
            query_type = "strategy"   
            
        elif any(term in question_lower for term in [
            "tell me about",
            "overview of",
            "summarize",
            "summary of",
            "company overview"
        ]):
            query_type = "executive_summary"     

        elif any(term in question_lower for term in [

            "revenue",
            "revenues",
            "income",
            "operating income",
            "net income",
            "profit",
            "profits",
            "profitability",
            "earnings",

            "financial",
            "financials",
            "financial highlight",
            "financial highlights",
            "financial performance",
            "financial results",

            "revenue drivers",
            "make money",
            "monetization",
            "business model"

        ]):
            query_type = "revenue"

        elif any(term in question_lower for term in [
            "growth",
            "growth opportunities",
            "opportunity",
            "opportunities",
            "expansion",
            "future growth",
            "future opportunities",
            "market opportunity",
            "market opportunities",
            "long term growth",
            "growth drivers"
        ]):
            query_type = "growth"

        elif any(term in question_lower for term in [
            "risk",
            "risks",
            "risk factor",
            "risk factors",
            "principal risk",
            "principal risks",
            "business risk",
            "business risks",
            "competitive risk",
            "competitive risks",
            "competition",
            "competitive",
            "cybersecurity",
            "security",
            "security incident",
            "security incidents",
            "regulation",
            "regulatory",
            "litigation",
            "legal",
            "legal proceeding",
            "legal proceedings",
            "compliance",
            "operational risk",
            "operational risks",
            "market risk",
            "market risks",
            "threat",
            "threats"
        ]):
            
            query_type = "risk"

        print(f"Query Type: {query_type}")
        
        is_comparison = any(
            word in question
            for word in [
                "compare",
                "vs",
                "versus",
                "difference",
                "different",

                "which company",
                "best positioned",
                "better positioned",
                "best investment",
                "long-term growth",
                "future growth",
                "growth potential",
                "strongest",
                "leader"
            ]
        )

        comparison_terms = []

        companies = [
            "amazon",
            "microsoft",
            "alphabet",
            "google",
            "meta",
            "apple",
            "nvidia",
            "tesla",
            "netflix",
            "adobe",
            "oracle",
            "salesforce",
            "amd",
            "intel",

            "aws",
            "azure",
            "microsoft cloud",
            "google cloud",

            "qualcomm",
            "walmart",
            "visa",
            "paypal",
            "cognizant",
            "accenture",

            "hcltech",
            "hcl tech",
            "hcl technologies",

            "ibm",
            "cisco",

            "servicenow",
            "service now",

            "uber"
        ]

        for company in companies:

            if company in question:
                comparison_terms.append(company)

        if "aws" in question_lower:
            comparison_terms.append("amazon")

        if "azure" in question_lower:
            comparison_terms.append("microsoft")

        # NEW CODE
        available_docs = [
            f.lower()
            for f in os.listdir("data/raw/business_docs")
        ]

        missing_companies = []
        
        company_aliases = {
            "hcltech": "hcl",
            "google": "alphabet",
            "google cloud": "alphabet",
            "aws": "amazon",
            "azure": "microsoft",
            "oracle cloud": "oracle",
            "facebook": "meta",
        }

        for company in comparison_terms:

            search_name = company_aliases.get(company, company)

            found = any(
                search_name in doc
                for doc in available_docs
            )

            if not found:
                missing_companies.append(company)
                
        if missing_companies:
            is_comparison = False
                   

        if len(comparison_terms) >= 2 and not missing_companies:
            is_comparison = True
             

        print(f"Comparison Query: {is_comparison}")
        print(f"Comparison Terms: {comparison_terms}")
        
        if is_comparison:
            COMPARISON_QUERIES.inc()
     
        if is_comparison:
            search_limit = 30
            
        elif query_type == "executive_summary":
            search_limit = 35    
            
        elif query_type in [
            "risk",
            "growth",
            "strategy",
            "segment",
            "revenue",
            "cloud",
            "ai",
            "executive_summary"
        ]:
            search_limit = 25  
            
        else:
            search_limit = 15

        print(f"Retrieval Limit: {search_limit}")

        # ----------------------------------
        # Search Qdrant
        # ----------------------------------

        print("\nSearching Qdrant...")
        search_start = time.time()

        if is_comparison and len(comparison_terms) >= 2:

            all_points = []

            for tech in comparison_terms:

                if query_type == "cloud":

                    tech_query = f"""
                    {tech}

                    cloud business
                    cloud revenue
                    cloud services
                    aws
                    azure
                    infrastructure
                    datacenter
                    compute
                    storage
                    database
                    cloud growth
                    """

                elif query_type == "ai":

                    if tech == "amazon":

                        tech_query = f"""
                        amazon

                        aws
                        amazon web services
                        bedrock
                        sagemaker
                        generative ai
                        machine learning
                        artificial intelligence
                        ai services
                        """

                    else:

                        tech_query = f"""
                        {tech}

                        artificial intelligence
                        ai strategy
                        machine learning
                        generative ai
                        copilot
                        foundation model
                        """
                    
                    
                    
                elif query_type == "segment":

                    tech_query = f"""
                    {tech}

                    business segments
                    operating segments
                    reportable segments
                    segment information
                    segment reporting
                    business units
                    organizational structure
                    productivity and business processes
                    intelligent cloud
                    more personal computing
                    """    

                elif query_type == "revenue":

                    tech_query = f"""
                    {tech}

                    revenue
                    operating income
                    profitability
                    sales growth
                    financial performance
                    """

                elif query_type == "growth":

                    tech_query = f"""
                    {tech}

                    growth opportunities
                    expansion
                    investments
                    future growth
                    market opportunity
                    """

                elif query_type == "risk":

                    tech_query = f"""
                    {tech}

                    risk factors
                    principal risks
                    business risks
                    competitive risks
                    competition
                    regulatory risks
                    regulation
                    cybersecurity risks
                    security incidents
                    litigation
                    legal proceedings
                    compliance
                    market risks
                    operational risks
                    threats
                    """

                else:

                    remaining_query = question.lower()

                    for company in comparison_terms:
                        remaining_query = remaining_query.replace(company, "")

                    remaining_query = remaining_query.replace("compare", "")
                    remaining_query = remaining_query.replace("and", "")

                    tech_query = f"{tech} {remaining_query}"
                
                print("\nTECH QUERY:")
                print(tech_query)

                tech_embedding = ollama_client.embeddings(
                    model="nomic-embed-text",
                    prompt=tech_query
                )["embedding"]

                tech_results = qdrant.query_points(
                    collection_name="business_docs",
                    query=tech_embedding,
                    limit=100
                )
                
                print(f"\nTOP SOURCES FOR {tech.upper()}")

                for point in tech_results.points[:10]:
                    print(point.payload.get("source"))

                print(
                    f"{tech}: {len(tech_results.points)} chunks retrieved"
                )

                company_points = []

                for point in tech_results.points:

                    source = point.payload.get(
                        "source",
                        ""
                    ).lower()

                    check_name = tech

                    aliases = {
                        "hcltech": "hcl",
                        "google": "alphabet",
                        "facebook": "meta",
                        "aws": "amazon",
                        "azure": "microsoft"
                    }

                    check_name = aliases.get(tech, tech)

                    if check_name in source:
                        company_points.append(point)

                    elif tech == "aws" and "amazon" in source:
                        company_points.append(point)

                    elif tech == "azure" and "microsoft" in source:
                        company_points.append(point)    
                        
                company_points = company_points[:20]     
                
                print(f"\n{tech.upper()} PAGES:")

                for point in company_points:
                    print(
                        point.payload.get("source"),
                        point.payload.get("page")
                    )

                print(
                    f"{tech}: {len(company_points)} chunks after source filtering"
                )

                all_points.extend(company_points)

            class TempResults:
                pass

            results = TempResults()
            results.points = all_points



        else:

            search_query = question
            
            if query_type == "general":

                search_query = f"""
                {question}

                company overview
                business overview

                revenue
                net income
                operating income
                profitability

                business segments
                operating segments

                revenue drivers

                growth opportunities
                future growth

                strategy
                long term vision

                investments
                capital expenditure
                research and development

                products
                services
                customers
                markets

                risk factors
                business risks
                """

            elif query_type == "segment":

                search_query = f"""
                {question}

                business segments
                operating segments
                reportable segments
                segment reporting
                business units
                organizational structure
                """
                
            elif query_type == "cloud":

                search_query = f"""
                {question}

                cloud business
                cloud platform
                cloud services
                cloud revenue
                cloud infrastructure
                datacenter
                hyperscale cloud
                cloud growth

                AWS
                Amazon Web Services
                Azure
                Google Cloud

                compute
                storage
                database
                analytics
                machine learning
                developers
                enterprises
                """  
                
            elif query_type == "ai":

                search_query = f"""
                {question}

                artificial intelligence
                ai strategy
                ai investments
                ai infrastructure
                machine learning
                generative ai
                genai
                llm
                foundation model
                large language model
                copilot
                gemini
                openai
                ai platform
                ai products
                ai services
                ai research
                ai innovation
                """    
                
            elif query_type == "executive_summary":

                search_query = f"""
                {question}

                company overview
                business overview
                business segments
                operating segments
                
                financial results
                revenue from operations
                total income
                profit before tax
                profit for the year
                tax expense
                earnings per share

                revenue
                revenue drivers
                profitability

                growth opportunities
                future growth

                artificial intelligence
                ai strategy

                risk factors
                business risks

                strategy
                long term vision

                products
                services
                customers
                markets
                """    

            query_embedding = ollama_client.embeddings(
                model="nomic-embed-text",
                prompt=search_query
            )["embedding"]
            
            if query_type == "executive_summary":
                search_limit = 400 

            elif query_type == "revenue":
                search_limit = 200

            elif query_type == "ai":
                search_limit = 200

            elif query_type == "strategy":
                search_limit = 200

            elif query_type == "growth":
                search_limit = 150

            elif query_type == "general":
                search_limit = 100
                   
            elif query_type == "cloud":
                search_limit = 50

            else:
                search_limit = 25


            results = qdrant.query_points(
                collection_name="business_docs",
                query=query_embedding,
                limit=search_limit
            )

        print("\nRetrieved Sources:")
        
        for point in results.points[:3]:
            print("\nPAYLOAD:")
            print(point.payload)

        for point in results.points:

            print(
                point.payload.get(
                    "source",
                    "Unknown"
                ),
                point.payload.get(
                    "page",
                    0
                )
            )

        print(
            f"Qdrant search took {time.time() - search_start:.2f} sec"
        )
        
        # ----------------------------------
        # Company Filtering
        # ----------------------------------

        company_name = detect_company(question)
        
        if company_name == "__UNKNOWN__":

            return {
                "question": request.question,
                "answer": """
        # Company Not Found

        The requested company is not available in the knowledge base.

        Please upload its annual report before querying.
        """,
                "sources": [],
                "confidence": 0,
                "quality": "Unavailable",
                "query_type": query_type,
                "company": ""
            }
        print("="*50)
        print(f"DETECTED COMPANY: {company_name}")
        print("="*50)

        if company_name == "hcltech":
            company_name = "hcl"

        # ==================================
        # VERIFY COMPANY EXISTS IN KB
        # ==================================

        if company_name:

            available_docs = [
                f.lower()
                for f in os.listdir("data/raw/business_docs")
            ]

            found = any(
                company_name in doc
                for doc in available_docs
            )

            if not found:

                return {
                    "question": request.question,
                    "answer": f"""
        # Company Not Found

        {company_name.upper()} is not available in the knowledge base.

        Please upload its annual report before querying.
        """,
                    "sources": [],
                    "confidence": 0,
                    "quality": "Unavailable",
                    "query_type": query_type,
                    "company": company_name
                }

        if is_comparison:
            print("\nComparison query detected - skipping company filtering")

        if company_name and not is_comparison:
            
            print(f"\nFiltering for company: {company_name}")
            print(f"Before Filter: {len(results.points)}")

            filtered_points = []

            for point in results.points:

                source = point.payload.get(
                    "source",
                    ""
                ).lower()

                if company_name in source:
                    filtered_points.append(point)

            results.points = filtered_points
            print(f"After Filter: {len(results.points)}")
            for point in results.points[:20]:
                print(
                    point.payload.get("source"),
                    point.payload.get("page")
                )
                
            financial_keywords = [
                "revenue",
                "net revenue",
                "net sales",
                "total revenue",

                "income from operations",
                "operating income",
                "operating profit",

                "family of apps",
                "reality labs",

                "segment information",
                "segment profitability",

                "financial results",
                "financial performance",

                "earnings per share",
                "eps",

                "cash flow",

                "research and development",
                "r&d",

                "capital expenditure",
                "capital expenditures",

                "total income",
                "profit before tax",
                "profit for the year"
            ]
            
            # ----------------------------------
            # Financial Chunk Prioritization
            # ----------------------------------

            if query_type == "executive_summary":

                financial_points = []
                other_points = []

                for point in results.points:

                    text = point.payload.get(
                        "text",
                        ""
                    ).lower()

                    if any(
                        keyword in text
                        for keyword in financial_keywords
                    ):
                        financial_points.append(point)

                    else:
                        other_points.append(point)

                results.points = (
                    financial_points +
                    other_points
                )

                print(
                    f"\nFinancial Chunks: {len(financial_points)}"
                )
                print(
                    f"Total Chunks: {len(results.points)}"
                )
            
            if query_type in [
                "revenue",
                "general",
                "executive_summary",
                "growth",
                "strategy"
            ]:


                results.points.sort(
                    key=lambda p: sum(
                        keyword in p.payload.get(
                            "text",
                            ""
                        ).lower()
                        for keyword in financial_keywords
                    ),
                    reverse=True
                )

            print(
                f"Remaining chunks after filtering: {len(results.points)}"
            )
            
            
        
        """# ----------------------------------
        # Hybrid Search Re-ranking
        # ----------------------------------"""

       

        
        # ----------------------------------
        # Build Context
        # ----------------------------------

        context = ""

        if is_comparison:
            comparison_context = {}

        sources = []
        seen_chunks = set()
        seen_sources = set()
        
        if is_comparison:
            MAX_CONTEXT_CHUNKS = 24
            
        elif query_type == "executive_summary":
            MAX_CONTEXT_CHUNKS = 20        

        elif query_type in [
            "risk",
            "growth",
            "strategy",
            "segment",
            "revenue",
            "cloud"
        ]:
            MAX_CONTEXT_CHUNKS = 12  

        else:
            MAX_CONTEXT_CHUNKS = 8
            
        context_chunks = 0

        for point in results.points:

            if not is_comparison and context_chunks >= MAX_CONTEXT_CHUNKS:
                break

            print("\nPAYLOAD:")
            print(point.payload)
            
            print(
                f"Score: {getattr(point, 'score', 0):.4f}"
            )

            source_name = point.payload.get(
                "source",
                "Unknown"
            )

            page_number = point.payload.get(
                "page",
                0
            )

            chunk_text = point.payload.get(
                "text",
                ""
            )

            # Clean common PDF extraction issues
            chunk_text = chunk_text.replace("/uni20B9", "₹")
            chunk_text = chunk_text.replace("po/r_t.ligafolio", "portfolio")
            chunk_text = chunk_text.replace("pa/r_t.liganer", "partner")
            
            # Skip extremely short chunks only
            if len(chunk_text.strip()) < 100:
                continue
            
            if is_comparison:

                matched_tech = None

                source_name_lower = source_name.lower()

                print("\nSOURCE NAME:")
                print(source_name_lower)
                
                print("\nCOMPARISON TERMS:")
                print(comparison_terms)

                for tech in comparison_terms:

                    check_name = tech

                    if tech == "hcltech":
                        check_name = "hcl"

                    print("CHECKING:", check_name)

                    if check_name in source_name_lower:

                        print("MATCHED:", check_name)

                        matched_tech = tech
                        break
                    
                if matched_tech:

                    if matched_tech not in comparison_context:
                        comparison_context[matched_tech] = []

                    comparison_context[matched_tech].append(chunk_text)
                    
                    print(
                        f"Added chunk to {matched_tech}"
                    )

            # ----------------------------------
            # Search Mode Filtering
            # ----------------------------------


            unique_key = chunk_text[:200]

            if unique_key in seen_chunks:
                continue

            seen_chunks.add(
                unique_key
            )

            context += (
                point.payload.get(
                    "text",
                    ""
                ) + "\n\n"
            )
            context_chunks += 1

            source_key = f"{source_name}_{page_number}"
            
            if source_key not in seen_sources:
                
                seen_sources.add(source_key)

                sources.append(
                    {
                        "source": source_name,
                        "page": page_number
                    }
                )
        print(f"\nFinal Context Chunks: {context_chunks}")
        
        if is_comparison:
            print("\nCOMPARISON CONTEXT:")

            for company, chunks in comparison_context.items():
                print(company, len(chunks))
        
        if is_comparison and comparison_context:

            context = ""

            for tech, chunks in comparison_context.items():

                context += f"\n\n### {tech.upper()}\n"

                unique_chunks = list(dict.fromkeys(chunks))

                for chunk in unique_chunks[:10]:

                    context += chunk + "\n\n"

        if context_chunks == 0:

            return {
                "answer": "No relevant documents found in the selected knowledge base.",
                "sources": [],
                "confidence": 0,
                "quality": "Low Quality",
                "response_time": round(
                    time.time() - start_time,
                    2
                )
            }                

        # ----------------------------------
        # Confidence Score
        # ----------------------------------

        scores = []

        for point in results.points[:5]:

            if hasattr(point, "score"):
                scores.append(point.score)
                
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score < 0.55:

            return {
                "answer": "No relevant information was found in the selected knowledge base for this question.",
                "sources": [],
                "confidence": round(avg_score * 100, 2),
                "quality": "Low Quality",
                "response_time": round(
                    time.time() - start_time,
                    2
                )
            }

        if scores:

            confidence = round(
                (sum(scores) / len(scores)) * 100,
                2
            )

        else:

            confidence = 0
            
        # Quality Badge

        if confidence >= 80:
            quality = "High Quality"

        elif confidence >= 60:
            quality = "Medium Quality"

        else:
            quality = "Low Quality"

        print(
            f"Confidence Score: {confidence}%"
        )

        print(
            f"Retrieved {len(sources)} unique sources"
        )

        # ----------------------------------
        # Prompt
        # ----------------------------------

        if is_comparison:

            if query_type == "cloud":

                prompt = f"""
                You are an Enterprise Cloud Business Analyst.

                IMPORTANT RULES:

                - Use ONLY the provided context.
                - Do NOT invent facts.
                - Extract exact numbers when available.
                - Prioritize cloud revenue, cloud growth, cloud products, cloud services.
                - If a number exists in the context, include it.

                Answer in the following format:

                ## Overview

                Give a short overview of each company's cloud business.

                ## Comparison Table

                | Company | Cloud Platform | Cloud Revenue | Growth Rate | Cloud Services | Key Cloud Investments |

                ## Key Differences

                - Compare AWS vs Azure.
                - Compare cloud revenue.
                - Compare cloud growth.
                - Compare cloud offerings.
                - Compare investments in cloud infrastructure.

                ## Final Assessment

                Based ONLY on the provided documents:

                - Which company has the larger cloud business?
                - Which company shows faster cloud growth?
                - Which company appears stronger in cloud infrastructure?

                Context:
                {context}

                Question:
                {request.question}

                Answer:
                """
            
            else:    

                prompt = f"""
                You are an Enterprise Business Analyst.

                IMPORTANT RULES:

                - Use ONLY the provided context.
                - Do NOT use outside knowledge.
                - Do NOT invent facts.
                - If information for a company is missing, explicitly say so.
                - Compare only what is available in the documents.

                Answer in the following format:

                ## Overview

                Give a short overview of each company.
                
                CRITICAL INSTRUCTION:

                If the question is about business segments:
                
                - List ALL business segments found in the context.
                - Do not stop after the first segment.
                - Combine information across multiple chunks.
                - Return every segment name that appears in the retrieved documents.
                - Extract the exact segment names from the context.
                - Copy the segment names exactly as written.
                - Do NOT summarize.
                - Do NOT paraphrase.
                - Do NOT replace segment names with business descriptions.
                - If segment names are not found, explicitly state that they are missing.

                ## Comparison Table

                | Company | Business Segments | Revenue Drivers | AI Strategy | Growth Opportunities | Key Risks |

                ## Key Differences

                - Explain the major differences between the companies.
                - If the question is about business segments, focus on segment structure first.
                - Use exact segment names from the documents.
                - Then discuss revenue, AI, growth, and risks if available.
                
                When comparing business segments:

                - Use exact segment names from the documents.
                - Never invent new segment categories.
                - Never replace official segment names with generic descriptions.

                ## Final Assessment

                Based ONLY on the provided documents:

                - Which company appears strongest?
                - Which company appears most AI-focused?
                - Which company appears to have the biggest growth opportunity?

                If information is missing, clearly state:

                "Not enough information available in retrieved documents."

                Context:
                {context}

                Question:
                {request.question}

                Answer:
                """

        elif query_type == "strategy":

            prompt = f"""
            You are an Enterprise Strategy Consultant.

            IMPORTANT RULES:

            - Use ONLY the provided context.
            - Do NOT use outside knowledge.
            - Do NOT invent facts.
            - Focus on business strategy, competitive positioning,
            investments, innovation, and long-term objectives.
            - ALWAYS include financial metrics when available.
            - Prefer exact numbers over generic descriptions.
            - If revenue, net income, operating income,
            growth rates, cash flow, AI investments,
            cloud investments, or R&D spending exist,
            include them in the answer.

            Answer in the following format:

            ## Strategy Overview

            Provide a short summary of the company's overall strategy.
            
            ## Financial Highlights

            - Revenue
            - Net Income
            - Operating Income
            - Growth Rates
            - R&D Investment

            Use exact numbers whenever available in the context.

            If unavailable say:

            "Information not available in retrieved documents."

            ## Strategic Priorities

            List the major strategic priorities.

            ## Competitive Positioning

            Explain how the company positions itself against competitors.

            ## AI and Technology Strategy

            Explain AI, cloud, platform, or technology initiatives.

            If not available, say:

            "Information not available in retrieved documents."

            ## Growth Strategy

            Explain how the company plans to grow.

            ## Strategic Investments

            Highlight major investments, acquisitions,
            partnerships, or innovation initiatives.

            ## Long-Term Vision

            Explain the company's future direction.

            ## Key Takeaways

            Provide 3-5 concise bullet points.

            Context:
            {context}

            Question:
            {request.question}

            Answer:
            """
            
        elif query_type == "risk":

            prompt = f"""
            You are an Enterprise Risk Analyst.

            IMPORTANT RULES:

            - Use ONLY the provided context.
            - Do NOT use outside knowledge.
            - Do NOT invent facts.
            - Focus ONLY on risks mentioned in the documents.
            - Ignore unrelated business information unless directly relevant to risks.

            Answer in the following format:

            ## Risk Overview

            Provide a short summary of the company's overall risk profile.

            ## Major Risks

            List the most important risks mentioned in the documents.

            ## Operational Risks

            Include supply chain, manufacturing, execution,
            workforce, infrastructure, or operational risks.

            If not available, say:
            "Information not available in retrieved documents."

            ## Financial Risks

            Include debt, interest rates, currency fluctuations,
            profitability risks, or financial uncertainty.

            If not available, say:
            "Information not available in retrieved documents."

            ## Regulatory Risks

            Include legal, compliance, tax,
            antitrust, privacy, or government risks.

            If not available, say:
            "Information not available in retrieved documents."

            ## Technology / Cybersecurity Risks

            Include cybersecurity, data privacy,
            AI risks, system failures, or technology risks.

            If not available, say:
            "Information not available in retrieved documents."

            ## Potential Business Impact

            Explain how these risks could affect the company.

            ## Key Takeaways

            Provide 3-5 concise bullet points.

            Context:
            {context}

            Question:
            {request.question}

            Answer:
            """   
            
        elif query_type == "growth":

            prompt = f"""
            You are an Enterprise Growth Strategy Analyst.

            IMPORTANT RULES:

            - Use ONLY the provided context.
            - Do NOT use outside knowledge.
            - Do NOT invent facts.
            - Focus ONLY on growth opportunities and future expansion.
            - ALWAYS include financial metrics when available.
            - Prefer exact numbers over generic descriptions.
            - If revenue, net income, operating income,
            growth rates, cash flow, AI investments,
            cloud investments, or R&D spending exist,
            include them in the answer.

            Answer in the following format:

            ## Growth Overview

            Provide a short summary of the company's growth strategy.
            
            ## Financial Highlights

            - Revenue
            - Net Income
            - Operating Income
            - Growth Rates
            - R&D Investment

            Use exact numbers whenever available in the context.

            If unavailable say:

            "Information not available in retrieved documents."

            ## Current Growth Drivers

            Explain the major drivers of current growth.

            ## AI Opportunities

            Explain AI-related growth opportunities.

            If not available, say:
            "Information not available in retrieved documents."

            ## Cloud Opportunities

            Explain cloud-related growth opportunities.

            If not available, say:
            "Information not available in retrieved documents."

            ## Expansion Opportunities

            Include new markets, international expansion,
            new products, acquisitions, or customer growth.

            ## Strategic Investments

            Highlight investments that support future growth.

            ## Long-Term Outlook

            Explain future growth potential based only on the documents.

            ## Key Takeaways

            Provide 3-5 concise bullet points.

            Context:
            {context}

            Question:
            {request.question}

            Answer:
            """  
            
        elif query_type == "segment":

            prompt = f"""
            You are an Enterprise Business Structure Analyst.

            IMPORTANT RULES:

            - Use ONLY the provided context.
            - Do NOT invent segment names.
            - Copy segment names exactly as written.
            - Combine segment information from all retrieved chunks.

            Answer in the following format:

            ## Business Segment Overview

            Brief explanation of how the company organizes its business.

            ## Business Segments

            List ALL segments exactly as written.

            ## Segment Description

            Explain what each segment does.

            ## Revenue Contribution

            Mention any revenue information available.

            ## Key Takeaways

            Provide 3-5 concise bullet points.

            Context:
            {context}

            Question:
            {request.question}

            Answer:
            """    
            
        elif query_type == "revenue":

            prompt = f"""
            You are an Enterprise Financial Analyst.

            IMPORTANT RULES:

            - Use ONLY the provided context.
            - Do NOT use outside knowledge.
            - Do NOT invent facts.
            - ALWAYS include exact financial numbers when available.
            - Prefer quantitative facts over generic descriptions.
            - If revenue, net income, operating income, growth rates,
            cash flow, R&D spending, or margins exist in the context,
            you MUST mention them.

            Answer in the following format:

            ## Revenue Overview

            Provide a short summary of how the company generates revenue.

            ## Financial Highlights

            - Revenue
            - Net Income
            - Operating Income
            - Growth Rates
            - R&D Investment

            Use exact numbers from the context whenever available.

            ## Main Revenue Sources

            List the company's major revenue streams.

            ## Revenue Drivers

            Explain the key factors driving revenue growth.

            ## High-Growth Areas

            Highlight business areas contributing most to growth.

            ## Strategic Investments

            Mention AI investments, cloud investments,
            infrastructure spending, acquisitions,
            or major growth investments.

            ## Revenue Risks

            Mention risks that could impact revenue.

            ## Key Takeaways

            Provide 3-5 concise bullet points.

            Context:
            {context}

            Question:
            {request.question}

            Answer:
            """ 
            
        elif query_type == "cloud":

            prompt = f"""
            You are an Enterprise Cloud Business Analyst.

            IMPORTANT RULES:

            - Use ONLY the provided context.
            - Do NOT use outside knowledge.
            - Do NOT invent facts.
            - Focus ONLY on cloud products, cloud services,
            cloud strategy, cloud growth, and cloud revenue.

            Answer in the following format:

            ## Cloud Overview

            Provide a short overview of the company's cloud business.

            ## Cloud Platform

            Identify the company's cloud platform and offerings.

            ## Cloud Services

            Explain major cloud products and services.

            ## Cloud Revenue Drivers

            Explain how the cloud business generates revenue.

            If not available, say:

            "Information not available in retrieved documents."

            ## Cloud Growth Opportunities

            Explain future growth opportunities for the cloud business.

            ## Cloud Investments

            Highlight cloud infrastructure, AI infrastructure,
            data centers, platform investments, or partnerships.

            ## Competitive Positioning

            Explain how the cloud business is positioned.

            ## Key Takeaways

            Provide 3-5 concise bullet points.

            Context:
            {context}

            Question:
            {request.question}

            Answer:
            """   
            
        elif query_type == "executive_summary":

            prompt = f"""
            You are an Enterprise Business Intelligence Analyst.

            IMPORTANT RULES:
            
            
            If exact numbers exist in the context,
            copy them exactly.

            Never rewrite financial figures.

            Never perform calculations.

            Never combine numbers from different sections
            unless explicitly stated in the context.

            Never guess.
            Never infer.
            Never use outside knowledge.
            - Use ONLY the provided context.
            - Do NOT use outside knowledge.
            - Do NOT invent facts.
            - If information is missing, explicitly say so.
            - Focus only on the company mentioned in the question.
            - ALWAYS include financial metrics when available.
            - Prefer exact numbers over generic descriptions.
            - If revenue, net income, operating income,
              growth rates, cash flow, AI investments,
              cloud investments, or R&D spending exist,
              include them in the answer.

            Answer in the following format:

            ## Company Overview

            Provide a concise overview using only information
            explicitly stated in the retrieved context.

            Do not use historical knowledge.
            Do not mention facts not found in context.
            
            ## Financial Highlights

            - Revenue
            - Net Income
            - Operating Income
            - Growth Rates
            - R&D Investment

            Use exact numbers whenever available in the context.

            If unavailable say:

            "Information not available in retrieved documents."

            ## Business Segments

            List ALL business segments exactly as written in the documents.

            ## Revenue Drivers

            Explain how the company generates revenue.

            ## Growth Opportunities

            Explain major growth opportunities mentioned in the documents.

            ## AI Strategy

            Summarize any AI-related information
            found in the retrieved context.
            If AI-related information exists anywhere in the context,
            this section must not be empty.

            Any mention of:

            - AI
            - Artificial Intelligence
            - AI-enabled
            - AI-powered
            - AI tools
            - AI platforms
            - AI products

            should be included in this section.

            Only return
            "Information not available in retrieved documents."
            if absolutely no AI-related text exists in the retrieved context.

            Do not invent AI investments or AI products.

            ## Key Risks

            Summarize major risks mentioned in the documents.

            ## Strategic Initiatives

            Highlight major strategic priorities,
            investments, acquisitions, innovation programs,
            cloud initiatives, or transformation efforts.

            ## Final Assessment

            Provide a short executive assessment of the company
            based ONLY on the retrieved documents.

            Context:
            {context}

            Question:
            {request.question}

            Answer:
            """        

        else:

            prompt = f"""
            You are an Enterprise Business Intelligence Assistant.

            IMPORTANT RULES:
            
            For company overview questions such as:

            - Tell me about Microsoft
            - Summarize Amazon
            - Explain Nvidia
            - Give me an overview of Apple
            
            If exact numbers exist in the context,
            copy them exactly.

            Never rewrite financial figures.

            Never perform calculations.

            Never combine numbers from different sections
            unless explicitly stated in the context.

            Only populate sections supported by the retrieved context.

            Do not invent information to fill missing sections.

            If a section has insufficient evidence in the context,
            write:

            "Information not available in retrieved documents."

            Never guess.
            Never infer.
            Never use outside knowledge.

            - Answer ONLY from the provided context.
            - Do NOT use outside knowledge.
            - Do NOT invent facts.
            - Do NOT make assumptions.

            - Focus on the company mentioned in the question.
            - Extract direct facts from the provided documents.
            - Prefer exact business information over generic descriptions.
            
            - For business segment questions:
                - List all business segments found in the context.
                - Use exact segment names from the documents.
                - Never invent segment names.
                - Never replace official segment names with business descriptions.
                - If official segments are found, copy them exactly as written.
                * Combine information across retrieved chunks.
                * Do not stop after the first segment.

            Answer in the following format:

            ## Company Overview

            Provide a concise summary using only facts explicitly stated in the retrieved context.

            Do not mention company history, founders, products, or business lines unless they appear in the retrieved context.  
            
            ## Financial Highlights

            Only include metrics explicitly found in the retrieved context.

            Do not calculate values.
            Do not estimate values.
            Do not infer missing values.
            If a metric is not explicitly present in the context,
            do not estimate it.

            Write:
            "Information not available in retrieved documents."

            Examples:
            - Revenue
            - Net Income
            - Operating Income
            - Cash Flow
            - R&D Spending

            Include only metrics actually present.

            Use exact numbers whenever available in the context.

            If unavailable say:

            "Information not available in retrieved documents."

            ## Business Segments

            - List all business segments found in the context.
            - Use exact segment names from the documents.
            - If multiple segments are found across different chunks,
              combine all segments into a single complete list.
            - If no segment information is available, say:
            "Business segment information not found."

            ## Revenue Drivers

            - Identify the main sources of revenue.
            - Include products, services, cloud businesses, subscriptions, advertising, hardware, software, or other revenue streams mentioned.

            ## Growth Opportunities

            - Explain major growth opportunities mentioned in the documents.
            - Include AI, cloud, international expansion, new products, acquisitions, or strategic investments if available.

            ## Key Risks

            - Summarize the most important risks mentioned.
            - Include operational, regulatory, competitive, cybersecurity, supply-chain, or financial risks when available.

            ## Strategic Initiatives

            - Highlight major strategic priorities.
            - Include AI initiatives, cloud investments, innovation programs, partnerships, sustainability initiatives, or transformation efforts.
            
            ## AI Strategy

            Summarize any AI-related information
            found in the retrieved context.
            If AI-related information exists anywhere in the context,
            this section must not be empty.

            Any mention of:

            - AI
            - Artificial Intelligence
            - AI-enabled
            - AI-powered
            - AI tools
            - AI platforms
            - AI products

            should be included in this section.

            Only return
            "Information not available in retrieved documents."
            if absolutely no AI-related text exists in the retrieved context.

            ## Key Takeaways

            Provide 3-5 concise bullet points summarizing the most important findings.

            If information for any section is not available in the retrieved documents, explicitly state:

            "Information not available in retrieved documents."

            If the answer is not clearly available in the context, respond exactly:

            I could not find enough information in the indexed documents to answer this question.

            Context:
            {context}

            Question:
            {request.question}

            Answer:
            """

        # ----------------------------------
        # LLM RESPONSE
        # ----------------------------------

        print("\nGenerating LLM response...")
        llm_start = time.time()
        
        print(f"Prompt Length: {len(prompt)}")
        
        response = ollama_client.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print(
            f"LLM took {time.time() - llm_start:.2f} sec"
        )

        answer = response["message"]["content"]

        # NEW CODE
        if missing_companies:

            return {
                "question": request.question,
                "answer": f"""
        ## Company Not Found

        The following companies are not available in the knowledge base:

        {', '.join(missing_companies)}

        Please upload their annual reports before requesting a comparison.
        """,
                "sources": [],
                "confidence": 0,
                "quality": "Unavailable",
                "query_type": query_type,
                "company": ""
            }


        if sources:

            answer += "\n\n---\n\n### References\n\n"

            for i, source in enumerate(
                sources,
                start=1
            ):

                source_name = source["source"]

                if source_name.startswith(
                    "http"
                ):

                    answer += (
                        f"[{i}] "
                        f"{source_name}\n"
                    )

                else:

                    filename = (
                        source_name
                        .split("/")[-1]
                    )

                    answer += (
                        f"[{i}] "
                        f"{filename} "
                        f"(Page {source['page']})\n"
                    )

        latency = time.time() - start_time
        
        REQUEST_LATENCY.observe(latency)
        
        print("Request completed successfully")
        
        conn = sqlite3.connect("logs/query_history.db")
          
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO query_history
        (timestamp, question, search_mode, response_time, confidence, quality)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            request.question,
            "Business Docs",
            round(latency, 2),
            confidence,
            quality
            ))
        conn.commit()
        conn.close()
        

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "confidence":confidence,
            "quality": quality,
            "query_type": query_type,
            "company": company_name,
            "is_comparison": is_comparison,
            "comparison_terms": comparison_terms
        }

    except Exception as e:

        print("\n==============================")
        print("ERROR OCCURRED")
        print("==============================")

        print(str(e))

        traceback.print_exc()

        return {
            "question": request.question,
            "answer": f"ERROR: {str(e)}",
            "sources": [],
            "confidence": 0
        }


# ==========================================
# PROMETHEUS METRICS ENDPOINT
# ==========================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )