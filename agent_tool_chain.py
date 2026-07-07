import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# Modern location for the Google Search utility
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler

# --- 1. Load Secrets from .env ---
load_dotenv()

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="Gemini AI Agent", page_icon="♊")
st.title("♊ Google Gemini Multi-Tool Agent")
st.write("Type a query that requires Google Search and math calculations.")


# --- 3. Define the Tools ---
@tool
def calculate_multiplier(number: float) -> float:
    """Multiplies a given number by 1.5. Use this for specific financial adjustments."""
    return number * 1.5


@tool
def google_search(query: str) -> str:
    """Searches Google for recent events, facts, or real-time web information."""
    # Automatically picks up GOOGLE_API_KEY and GOOGLE_CSE_ID from environment
    search = GoogleSearchAPIWrapper()
    return search.run(query)


tools = [google_search, calculate_multiplier]

# --- 4. Main Application Logic ---
user_query = st.text_input(
    "Enter your query:",
    placeholder="e.g., Find the stock price of Alphabet (GOOGL) today and multiply it by 1.5.",
)

if st.button("Run Agent"):
    if not user_query:
        st.warning("Please type a query first.")
    else:
        try:
            # Initialize Google Gemini Model
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

            # The modern API returns a runnable graph executor object directly
            agent_executor = create_agent(model=llm, tools=tools)

            # Create a UI container for execution logs
            with st.status(
                "Gemini is thinking and executing tools...", expanded=True
            ) as status:
                st_callback = StreamlitCallbackHandler(st.container())

                # Execute using standard message array formatting
                response = agent_executor.invoke(
                    {"messages": [{"role": "user", "content": user_query}]},
                    {"callbacks": [st_callback]},
                )
                status.update(
                    label="Execution complete!", state="complete", expanded=False
                )

            # Display final answer (Modern response object holds data in a messages array)
            st.subheader("Final Output:")
            st.success(response["messages"][-1].content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
