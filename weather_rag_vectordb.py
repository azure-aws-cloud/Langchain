import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Core LangChain Tools & Messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Google Gemini Integration
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Standalone Partner Integrations (No langchain-community used)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# 🛠️ ENV LOADING & SETUP
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("🚀 SYSTEM START: Initialising Chroma RAG + Tool Environment")
print("=" * 50)

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print("✅ Success: GOOGLE_API_KEY discovered.")
else:
    print("❌ Error: GOOGLE_API_KEY was NOT found in your .env file.")
print("=" * 50 + "\n")

MD_FILE_PATH = "knowledge_base.md"


# ---------------------------------------------------------
# 📚 1. MARKDOWN INITIALIZATION & CHROMA DB SETUP
# ---------------------------------------------------------
def ensure_markdown_file():
    """Writes the knowledge base to a .md file if it does not exist."""
    if not os.path.exists(MD_FILE_PATH):
        print(f"📝 [File IO] '{MD_FILE_PATH}' not found. Generating default knowledge base...")
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
        print(f"✅ [File IO] Default '{MD_FILE_PATH}' file written successfully.")
    else:
        print(f"📖 [File IO] Existing '{MD_FILE_PATH}' discovered.")


@st.cache_resource
def initialize_chroma_db():
    """Reads the .md file and initializes the standalone langchain-chroma Vector DB."""
    ensure_markdown_file()

    print(f"📖 [RAG Setup] Loading content from local '{MD_FILE_PATH}'...")
    with open(MD_FILE_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = text_splitter.split_text(md_text)

    # Initialize native Google embeddings
    print("🧠 [RAG Setup] Generating vector embeddings via Google Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

    # Instantiate vector store directly from langchain_chroma
    print("💎 [RAG Setup] Indexing chunks inside Chroma DB collection...")
    chroma_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="travel_guidelines"
    )
    print("✅ [RAG Setup] Chroma Vector Database built successfully.")
    return chroma_store


# ---------------------------------------------------------
# 🔧 2. DEFINE LLM TOOLS (Live API + Chroma DB Retrieval)
# ---------------------------------------------------------
@tool
def get_current_weather(location: str) -> str:
    """Fetch the current live weather matrices for a given city location using a free weather API."""
    print(f"\n[🛠️ Live API Tool] Executing weather check for: '{location}'")
    location_cleaned = str(location).strip()

    # geo_url = f"https://open-meteo.com{location_cleaned}&count=1&language=en&format=json"
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_cleaned}&count=1&language=en&format=json"

    try:
        geo_res = requests.get(geo_url, timeout=15).json()
        results_list = geo_res.get("results")

        if not results_list or len(results_list) == 0:
            return f"Could not find coordinates for {location_cleaned}."

        first_match = results_list[0]
        lat = first_match.get("latitude")
        lon = first_match.get("longitude")

        # weather_url = f"https://open-meteo.com{lat}&longitude={lon}&current_weather=true"
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_res = requests.get(weather_url, timeout=15).json()
        current = weather_res.get("current_weather", {})

        temp = current.get("temperature", "unknown")
        wind = current.get("windspeed", "unknown")

        output_str = f"The current temperature in {location_cleaned} is {temp}°C with a wind speed of {wind} km/h."
        print(f"📦 [Live API Tool] Output payload generated: '{output_str}'")
        return output_str
    except Exception as e:
        return f"Error retrieving live weather data: {str(e)}"


@tool
def search_local_travel_guidelines(query: str) -> str:
    """Search internal unstructured document manual guides for local tips, safety guidelines, and packing recommendations."""
    print(f"\n[📚 RAG Tool] Searching Chroma Vector DB for query context: '{query}'")

    # Safely reference cached Chroma database instance from state
    db = st.session_state.chroma_db

    # Perform similarity search
    search_results = db.similarity_search(query, k=2)

    retrieved_context = "\n".join([doc.page_content for doc in search_results])
    print(f"📦 [RAG Tool] Relevant matching text extracted from Chroma:\n{retrieved_context}")
    return f"Retrieved Context from internal guidelines:\n{retrieved_context}"


# ---------------------------------------------------------
# 🖥️ 3. STREAMLIT UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Gemini Standalone Chroma App", page_icon="🤖")
st.title("🌤️ Gemini Native Chroma RAG + Weather Assistant")
st.write("Ask Gemini about weather and markdown-grounded travel guidelines.")

if api_key:
    # Initialize Chroma database into session state
    if "chroma_db" not in st.session_state:
        with st.spinner("Processing Markdown and building Chroma DB..."):
            st.session_state.chroma_db = initialize_chroma_db()

    # Initialize Gemini 2.5 Flash LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    tools_list = [get_current_weather, search_local_travel_guidelines]
    llm_with_tools = llm.bind_tools(tools_list)

    user_query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., Check the weather in Paris and find out if there are any special guidelines."
    )

    if user_query:
        print(f"\n[📥 Input UI] User submitted query packet: '{user_query}'")
        status_box = st.empty()

        conversation_history = [HumanMessage(content=user_query)]

        try:
            status_box.info("🧠 step 1: Asking Gemini to reason through the query...")
            initial_response = llm_with_tools.invoke(user_query)

            if initial_response.tool_calls:
                print(f"🤖 [Model Strategy] Gemini generated tool calls: {initial_response.tool_calls}")
                conversation_history.append(initial_response)

                for tool_call in initial_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    status_box.info(f"⚙️ step 2: Running Tool: `{tool_name}`...")

                    if tool_name == "get_current_weather":
                        tool_output = get_current_weather.invoke(tool_args)
                    elif tool_name == "search_local_travel_guidelines":
                        tool_output = search_local_travel_guidelines.invoke(tool_args)
                    else:
                        tool_output = "Error: Unrecognised tool mapping structure call."

                    conversation_history.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )

                status_box.info("📝 step 3: Generating final blended summary response...")
                print("🧠 [Processing] Submitting entire context chain history back to Gemini...")
                final_response = llm.invoke(conversation_history)

                status_box.empty()
                st.subheader("🤖 Response:")
                st.write(final_response.content)
            else:
                print("🤖 [Model Strategy] Direct text response returned without tool requirements.")
                status_box.empty()
                st.subheader("🤖 Response:")
                st.write(initial_response.content)

        except Exception as e:
            status_box.empty()
            st.error(f"Execution Error encountered: `{str(e)}`")
            print(f"💥 [Runtime System Crash] Critical fault: {str(e)}")
else:
    st.warning("Please add your API key to the `.env` file to begin.")
