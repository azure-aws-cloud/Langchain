import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------
# 🛠️ ENV LOADING & CONSOLE DEBUGGING
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("🚀 SYSTEM START: Initialising Environment Setup")
print("=" * 50)

# Load environment variables from .env
env_loaded = load_dotenv(override=True)
print(f"📡 Debug: load_dotenv() executed -> Status: {env_loaded}")

api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"✅ Success: GOOGLE_API_KEY discovered.")
    print(f"🔑 Key Preview: {api_key[:6]}...{api_key[-4:]} (Length: {len(api_key)})")
else:
    print("❌ Error: GOOGLE_API_KEY was NOT found in your .env file.")
print("=" * 50 + "\n")


# 1. Weather API Tool with exact URL structures
@tool
def get_current_weather(location: str) -> str:
    """Fetch the current weather for a given city location using a free weather API."""
    print(f"\n[🛠️ Tool Call] Executing 'get_current_weather' for: '{location}'")

    # Clean up trailing whitespaces from the city argument
    location_cleaned = str(location).strip()

    # ✅ FIXED: Correct subdomain and path for Geocoding
    # https://geocoding-api.open-meteo.com/v1/search?name=Paris&count=1&language=en&format=json
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_cleaned}&count=1&language=en&format=json"
    print(f"🔗 [Tool] Requesting Geolocation data from: {geo_url}")

    try:
        # ✅ INCREASED TIMEOUT: Bumped from 5 to 15 to handle slow network responses safely
        geo_res = requests.get(geo_url, timeout=15).json()
        results_list = geo_res.get("results")

        # If the API returns no results array or it's empty
        if not results_list or len(results_list) == 0:
            print(
                f"⚠️ [Tool] No coordinates found for location target: '{location_cleaned}'"
            )
            return f"Could not find coordinates for {location_cleaned}."

        # Access index 0 since results is a list of matching cities
        first_match = results_list[0]
        lat = first_match.get("latitude")  # 48.8534
        lon = first_match.get("longitude")  # 2.3488
        print(f"🎯 [Tool] Found Coordinates: Latitude={lat}, Longitude={lon}")

        # ✅ FIXED BOTH PATH & PARAMETERS: Correct domain, added explicit 'latitude=' query key string
        # https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        print(f"🔗 [Tool] Requesting current weather matrices from: {weather_url}")

        # ✅ INCREASED TIMEOUT: Bumped to 15 seconds to prevent connection dropping
        weather_res = requests.get(weather_url, timeout=30)
        print(f"Results from Weather API is  ", weather_res)
        weather_data = weather_res.json()
        print(f"Results from Weather API in JSON   ", weather_data)
        current = weather_data.get("current_weather", {})

        temp = current.get("temperature", "unknown")
        wind = current.get("windspeed", "unknown")

        output_str = f"The current temperature in {location_cleaned} is {temp}°C with a wind speed of {wind} km/h."
        print(f"📦 [Tool] Output payload generated: '{output_str}'")
        return output_str

    except Exception as e:
        print(f"the exception is ", e)
        print(f"💥 [Tool Exception] Error during API request: {str(e)}")
        return f"Error retrieving weather data: {str(e)}"


# 2. Streamlit UI Setup
st.set_page_config(page_title="Gemini Weather Assistant", page_icon="🌤️")
st.title("🌤️ Gemini AI Weather Assistant")
st.write("Ask Gemini about the weather anywhere in the world!")

# --- UI SIDEBAR DIAGNOSTICS ---
with st.sidebar:
    st.header("⚙️ Diagnostics")
    if api_key:
        st.success(f"API Key Loaded")
        st.caption(f"Model In Use: `gemini-2.5-flash`")
        st.caption(f"Starts with: `{api_key[:6]}...`")
    else:
        st.error("❌ GOOGLE_API_KEY is missing from your .env file!")
        st.info("Ensure your `.env` file looks like this:\n`GOOGLE_API_KEY=AIzaSy...`")

# 3. Main Logic Execution Loop
if api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=api_key, temperature=0
    )

    tools = [get_current_weather]
    llm_with_tools = llm.bind_tools(tools)

    user_query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What is the weather like in Paris right now?",
    )

    if user_query:
        print(f"\n[📥 Input UI] Received user query: '{user_query}'")
        status_box = st.empty()

        try:
            print("🧠 [Processing] step 1: Dispatching query to gemini-2.5-flash...")
            status_box.info("🧠 step 1: Querying Gemini Model...")
            response = llm_with_tools.invoke(user_query)

            if response.tool_calls:
                print(
                    f"🤖 [Model Response] Gemini triggered tool calls: {response.tool_calls}"
                )
                for tool_call in response.tool_calls:
                    city = tool_call["args"].get("location")

                    print(
                        f"🌐 [Processing] step 2: Executing weather tool code for: '{city}'..."
                    )
                    status_box.info(
                        f"🌐 step 2: Calling Weather API Tool for '{city}'..."
                    )
                    tool_output = get_current_weather.invoke(tool_call["args"])

                    print("📝 [Processing] step 3: Constructing message context...")
                    status_box.info(
                        "📝 step 3: Compiling structured message history for summary..."
                    )

                    conversation_history = [
                        HumanMessage(content=user_query),
                        AIMessage(content="", tool_calls=response.tool_calls),
                        ToolMessage(
                            content=str(tool_output), tool_call_id=tool_call["id"]
                        ),
                    ]

                    print(
                        "🧠 [Processing] Requesting final user-friendly summary from Gemini..."
                    )
                    final_response = llm.invoke(conversation_history)

                    print(f"✨ [Final Response] Process completed successfully.")
                    status_box.empty()
                    st.subheader("☀️ Response:")
                    st.write(final_response.content)
            else:
                print(
                    "🤖 [Model Response] Gemini answered directly without needing the weather tool."
                )
                status_box.empty()
                st.subheader("☀️ Response:")
                st.write(response.content)

        except Exception as e:
            print(f"💥 [Runtime System Crash] Critical fault: {str(e)}")
            status_box.empty()
            st.error(f"Execution Error encountered: `{str(e)}`")
else:
    st.warning("Please add your API key to the `.env` file to begin.")
