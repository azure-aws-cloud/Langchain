import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Added for CORS support
from pydantic import BaseModel
from dotenv import load_dotenv

# Core LangChain Components
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialization and Env Setup
load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("CRITICAL: GOOGLE_API_KEY was NOT found in your .env file.")

app = FastAPI(title="Gemini Hybrid RAG + Tool API Engine")

# ---------------------------------------------------------
# 🛡️ CORS MIDDLEWARE SETUP
# ---------------------------------------------------------
# Allows browser cross-origin requests from any frontend port (e.g. Streamlit, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

MD_FILE_PATH = "knowledge_base.md"
CHROMA_STORE = None


def ensure_markdown_file():
    """Writes the knowledge base to a .md file if it does not exist."""
    if not os.path.exists(MD_FILE_PATH):
        markdown_content = """# Global Travel Guidelines & Advisories
## Paris Travel Manual
Paris Travel Guideline: When temperatures drop below 15°C, local tourist open-top buses offer free heated blankets to passengers.
During heavy wind speeds exceeding 20 km/h, the top deck of the Eiffel Tower may close temporarily for safety.

## Tokyo Travel Manual
Tokyo Travel Guide: Summer months are highly humid; carrying portable misting fans is recommended.

## London Travel Manual
London Advisory: Always carry a compact umbrella regardless of the morning temperature prediction.
"""
        with open(MD_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(markdown_content)


def initialize_chroma_db():
    """Reads the .md file and initializes the standalone langchain-chroma Vector DB."""
    global CHROMA_STORE
    ensure_markdown_file()

    with open(MD_FILE_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = text_splitter.split_text(md_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

    CHROMA_STORE = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="travel_guidelines"
    )
    print("📡 [Backend Boot] Chroma Vector Database built successfully.")


# Trigger vector DB indexing immediately upon API application start
@app.on_event("startup")
def startup_event():
    initialize_chroma_db()


# ---------------------------------------------------------
# 🏥 HEALTH CHECK ENDPOINT
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    """Returns the operational status of the server engine and connected systems."""
    is_database_ready = CHROMA_STORE is not None
    status_code = "healthy" if is_database_ready else "degraded"

    return {
        "status": status_code,
        "database_connected": is_database_ready,
        "api_key_loaded": api_key is not None
    }


# 2. Define Core LLM Tools
@tool
def get_current_weather(location: str) -> str:
    """Fetch the current live weather matrices for a given city location using a free weather API."""
    location_cleaned = str(location).strip()
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_cleaned}&count=1&language=en&format=json"
    try:
        geo_res = requests.get(geo_url, timeout=15).json()
        results_list = geo_res.get("results")
        if not results_list or len(results_list) == 0:
            return f"Could not find coordinates for {location_cleaned}."

        first_match = results_list[0]
        lat = first_match.get("latitude")
        lon = first_match.get("longitude")

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_res = requests.get(weather_url, timeout=15).json()
        current = weather_res.get("current_weather", {})

        temp = current.get("temperature", "unknown")
        wind = current.get("windspeed", "unknown")
        return f"The current temperature in {location_cleaned} is {temp}°C with a wind speed of {wind} km/h."
    except Exception as e:
        return f"Error retrieving live weather data: {str(e)}"


@tool
def search_local_travel_guidelines(query: str) -> str:
    """Search internal unstructured document manual guides for local tips, safety guidelines, and packing recommendations."""
    if CHROMA_STORE is None:
        return "Error: Vector store is not ready."
    search_results = CHROMA_STORE.similarity_search(query, k=2)
    retrieved_context = "\n".join([doc.page_content for doc in search_results])
    return f"Retrieved Context from internal guidelines:\n{retrieved_context}"


# 3. REST Schema Endpoints
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    steps: list[str]


@app.post("/api/chat", response_model=QueryResponse)
def handle_chat_query(request: QueryRequest):
    steps_taken = []
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
        tools_list = [get_current_weather, search_local_travel_guidelines]
        llm_with_tools = llm.bind_tools(tools_list)

        steps_taken.append("🧠 Querying Gemini model for intent...")
        initial_response = llm_with_tools.invoke(request.query)

        if initial_response.tool_calls:
            conversation_history = [HumanMessage(content=request.query), initial_response]

            for tool_call in initial_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                steps_taken.append(f"⚙️ Running Tool: `{tool_name}` for context arguments -> {tool_args}")

                if tool_name == "get_current_weather":
                    tool_output = get_current_weather.invoke(tool_args)
                elif tool_name == "search_local_travel_guidelines":
                    tool_output = search_local_travel_guidelines.invoke(tool_args)
                else:
                    tool_output = "Error: Unrecognised tool mapping structure call."

                conversation_history.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                )

            steps_taken.append("📝 Blending tool payload histories for unified agent response...")
            final_response = llm.invoke(conversation_history)
            return QueryResponse(response=final_response.content, steps=steps_taken)
        else:
            steps_taken.append("🤖 Gemini responded directly without requiring external tool routing paths.")
            return QueryResponse(response=initial_response.content, steps=steps_taken)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
