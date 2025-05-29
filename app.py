from fastapi import FastAPI, Query
from pydantic import BaseModel
import json
import re
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer
import os
from typing import Optional
# Load from .envs

# Read env variables
GEMINI_API_KEY='AIzaSyC01IfS8b9nxxSNJupKNL35zUOisTbRBvE'
COLLECTION_NAME='Growcite_candidate_final'
QDRANT_HOST='localhost'
QDRANT_PORT=6333

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.0-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(
    title="Growcite Candidate Search Tool",
    description="A semantic + structured filter search tool for Growcite's candidate database.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class PaginatedResponse(BaseModel):
    results: list
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

# Define role synonyms
ROLE_SYNONYMS = {
    "dot net developer": [".net developer", "dotnet developer", "dot net dev", ".net dev", "dot net developers"],
    "data scientist": ["data science specialist", "ml engineer", "machine learning expert"],
    "frontend developer": ["frontend dev", "frontend engineer", "front end developer", "react developer"],
    "backend developer": ["backend dev", "back end developer", "java developer", "nodejs developer"],
    # Add more mappings as needed
}

def normalize_role(role_value: str) -> str:
    role_value = role_value.lower()
    for canonical, synonyms in ROLE_SYNONYMS.items():
        if role_value in [s.lower() for s in synonyms] or role_value == canonical:
            return canonical
    return role_value

def extract_filters(prompt):
    llm_prompt = f"""
    You are a helpful assistant that extracts structured filters from natural language queries.

    Candidate fields you can match on:
    - candidate_id (exact match, string like "candidate_1882")
    - name (exact match)
    - role (string match)
    - location (string match)
    - experience (number, in years)
    - current_ctc (number, in LPA)
    - expected_ctc (number, in LPA)
    - notice_period (number, in days or months)
    - education (string match)
    - status (string match)
    - email (exact match)
    - phone (exact match)

    Query:
    "{prompt}"

    Return only JSON in this format:
    {{
      "filters": [
        {{
          "field": "role",
          "type": "match",
          "value": "MIS Analyst"
        }},
        {{
          "field": "location",
          "type": "match",
          "value": "Mumbai"
        }},
        {{
          "field": "experience",
          "type": "range",
          "gte": 0,
          "lte": 1
        }}
      ]
    }}

    Rules:
    - Use `"match"` for exact or partial string matches.
    - Use `"range"` for numeric values (experience, CTC, notice period).
    - Convert units like "months", "LPA", "Lakhs" into just numbers.
    - Only return the JSON. Do not explain anything.
    """
    response = llm_model.generate_content(llm_prompt)
    json_block = re.search(r'\{[\s\S]*\}', response.text)
    if not json_block:
        return []
    return json.loads(json_block.group())["filters"]

def build_qdrant_filter(conditions_json: list) -> Filter:
    must_conditions = []
    for condition in conditions_json:
        field = condition["field"]
        condition_type = condition["type"]

        if condition_type == "match":
            value = condition["value"]
            if field == "role":
                value = normalize_role(value)
            must_conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))

        elif condition_type == "range":
            range_kwargs = {}
            if "gte" in condition:
                range_kwargs["gte"] = condition["gte"]
            if "lte" in condition:
                range_kwargs["lte"] = condition["lte"]
            must_conditions.append(FieldCondition(key=field, range=Range(**range_kwargs)))

    return Filter(must=must_conditions)

def paginate_results(results: list, page: int, page_size: int) -> dict:
    """
    Paginate a list of results and return pagination metadata
    """
    total_count = len(results)
    total_pages = (total_count + page_size - 1) // page_size  # Ceiling division
    
    # Calculate start and end indices
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    # Get the paginated results
    paginated_results = results[start_idx:end_idx]
    
    return {
        "results": paginated_results,
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }

@app.post("/search")
def search_candidates(
    request: QueryRequest,
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of results per page (max 100)")
):
    """
    Search candidates with pagination support
    
    - **query**: Natural language search query
    - **page**: Page number (starting from 1)
    - **page_size**: Number of results per page (1-100)
    """
    filters = extract_filters(request.query.lower())
    vector = embedder.encode(request.query.lower()).tolist()
    qdrant_filter = build_qdrant_filter(filters)

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("Built Qdrant Filter:", qdrant_filter)

    # Get all results first (you might want to increase this limit based on your needs)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=1000,  # This is the maximum we fetch from Qdrant
        query_filter=qdrant_filter,
        with_payload=True,
        with_vectors=True,
    )

    print("Total Results from Qdrant:", len(results))

    # Format all results
    formatted_results = [
        {
            "name": res.payload.get("name"),
            "role": res.payload.get("role"),
            "education": res.payload.get("education"),
            "location": res.payload.get("location"),
            "status": res.payload.get("status"),
            "experience": res.payload.get("experience"),
            "notice_period": res.payload.get("notice_period"),
            "current_ctc": res.payload.get("current_ctc"),
            "expected_ctc": res.payload.get("expected_ctc"),
            "email": res.payload.get("email"),
            "phone": res.payload.get("phone"),
        }
        for res in results
    ]

    # Apply pagination
    paginated_response = paginate_results(formatted_results, page, page_size)
    
    return paginated_response

@app.get("/search")
def search_candidates_get(
    query: str = Query(..., description="Natural language search query"),
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of results per page (max 100)")
):
    """
    Alternative GET endpoint for search with pagination
    
    - **query**: Natural language search query
    - **page**: Page number (starting from 1)
    - **page_size**: Number of results per page (1-100)
    """
    request = QueryRequest(query=query)
    return search_candidates(request, page, page_size)

@app.get("/")
def root():
    return {
        "message": "Growcite Candidate Search Tool",
        "endpoints": {
            "POST /search": "Search candidates with JSON body",
            "GET /search": "Search candidates with query parameters",
        },
        "pagination": {
            "default_page_size": 10,
            "max_page_size": 100,
            "page_numbering": "starts from 1"
        }
    }