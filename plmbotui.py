import streamlit as st
import requests

# ---------------------------------------------------------
# 🛠️ SYSTEM TERMINAL LOGGING BANNER
# ---------------------------------------------------------
# ---------------------------------------------------------
# 🛠️ SYSTEM TERMINAL LOGGING BANNER
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("🚀 FRONTEND START: Initialising 3DExperience Application ")
print("=" * 50)

# Updated the browser tab favicon to use the cloud icon as well
st.set_page_config(page_title="Welcome to PLMBOT for 3DExperience", page_icon="☁️")

# REPLACED: Title modified with corporate 3DExperience Cloud mapping
st.title("☁️ 3DExperience PLMBOT Agent")

# 🛰️ TARGET SERVER CONFIGURATION
BACKEND_HOST = "http://127.0.0.1:8000"
CHAT_URL = f"{BACKEND_HOST}/api/chat"
HEALTH_URL = f"{BACKEND_HOST}/health"

# --- SESSION STATE INITIALISATION ---
# Holds the array of previous unique user queries
if "history" not in st.session_state:
    st.session_state.history = []

# Tracks the query to be automatically executed in the main loop
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

# --- SIDEBAR HISTORY COMPONENT ---
st.sidebar.title("📜 Command History")

if st.session_state.history:
    st.sidebar.caption("Click a past command to re-run it:")
    for idx, past_cmd in enumerate(st.session_state.history):
        # Truncate string for sidebar view button safety
        short_label = past_cmd[:40] + "..." if len(past_cmd) > 40 else past_cmd

        # When a button is clicked, it updates active_query and forces a rerun
        if st.sidebar.button(f"💬 {short_label}", key=f"hist_{idx}"):
            st.session_state.active_query = past_cmd
            st.rerun()
else:
    st.sidebar.info("No past commands yet.")

st.sidebar.markdown("---")

# --- AUTOMATIC HEALTH ROUTINE CHECK ---
backend_online = False

print(f"📡 Debug: Initiating health check handshake request to -> {HEALTH_URL}")
try:
    health_response = requests.get(HEALTH_URL, timeout=3)
    print(
        f"📡 Debug: Server responded with HTTP status code -> {health_response.status_code}"
    )

    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f"📡 Debug: JSON Payload payload parsed successfully -> {health_data}")

        if health_data.get("status") == "healthy":
            backend_online = True
            st.sidebar.success("🟢 PLMBOT : Connected")
            st.sidebar.caption(f"Host: `{BACKEND_HOST}`")
            print("✅ Handshake verified: Backend is alive and Chroma is connected.")
        else:
            st.sidebar.warning("🟡 Backend Status: Degraded (Check Vector DB)")
            print(
                "⚠️ Handshake warning: Backend responded but database object is missing."
            )
    else:
        st.sidebar.error(f"🔴 Backend Error Code: {health_response.status_code}")
except requests.exceptions.RequestException as e:
    st.sidebar.error("🔴 Backend Status: Offline / Unreachable")
    print(f"💥 Network Error: Handshake completely failed. Exception details: {str(e)}")

print("=" * 50 + "\n")

# --- UI LOGIC INTERACTION ---
if backend_online:
    # Use value=st.session_state.active_query to populate text when sidebar is clicked
    user_query = st.text_input(
        "What would you like to ask?",
        value=st.session_state.active_query,
        placeholder="e.g., Unlock Engineering Change Action Axle 1.9 or Transfer ownership...",
    )

    # If the user typed something fresh, update active_query to match it
    if user_query and user_query != st.session_state.active_query:
        st.session_state.active_query = user_query

    # Run execution pipeline if active_query is set
    if st.session_state.active_query:
        current_execution_query = st.session_state.active_query

        # Save to history list if it's new
        if current_execution_query not in st.session_state.history:
            st.session_state.history.append(current_execution_query)
            st.rerun()  # Rerun once to visually inject the item into sidebar state safely

        print(
            f"\n[📥 UI Action] User submitted text query payload: '{current_execution_query}'"
        )

        # UI Visual Debugger Container
        debug_container = st.expander(
            "🔍 Live UI Execution Trace Flow Logs", expanded=True
        )

        with st.spinner("Streaming data packets from REST backend endpoint..."):
            try:
                # Trace Step 1
                debug_container.info(
                    "🔹 **Step 1:** Packaging request body parameters into a secure JSON packet structure..."
                )
                payload = {"query": current_execution_query}
                print(f"📦 Payload structural dictionary created: {payload}")

                # Trace Step 2
                debug_container.info(
                    f"🔹 **Step 2:** Despatching HTTP POST transport request packet to endpoint: `{CHAT_URL}`..."
                )
                print(f"🚀 Transmitting HTTP POST transaction to REST backend host...")
                response = requests.post(CHAT_URL, json=payload, timeout=30)

                print(
                    f"📥 Received transaction answer status from backend -> HTTP {response.status_code}"
                )

                if response.status_code == 200:
                    # Trace Step 3
                    debug_container.info(
                        "🔹 **Step 3:** Parsing incoming compressed execution steps and response strings..."
                    )
                    data = response.json()
                    print(
                        f"✨ Successfully parsed server metrics response payload dictionary key count: {len(data)}"
                    )

                    # Clean visual logs and present output data maps cleanly
                    debug_container.success(
                        "🔹 **Step 4:** Execution loop completed successfully! Rendering outputs below..."
                    )

                    # Render the internal execution steps safely out to the user UI
                    st.subheader("⚙️ Agent Execution Log Steps")
                    for step in data.get("steps", []):
                        st.caption(step)

                    # Print the final synthesized answer string response block
                    st.subheader("🤖 Agent Answer Response:")
                    st.write(data.get("response"))
                else:
                    debug_container.error(
                        f"💥 Flow Interrupted: Backend returned status payload code `{response.status_code}`"
                    )
                    st.error(
                        f"Backend Server returned an Error Code: `{response.status_code}`"
                    )
                    st.info(response.text)

            except Exception as e:
                print(f"💥 Unexpected frontend workflow hazard caught: {str(e)}")
                st.error(f"An unexpected framework hazard occurred: {str(e)}")
else:
    # Error message displayed if the health check fails
    st.error("❌ Cannot establish connection to the backend REST API engine.")
    st.warning(
        f"Ensure that your PLMBOT is running on **{BACKEND_HOST}** before launching this UI."
    )
    st.info(
        "💡 Tip: Try opening http://127.0.0:8000/health in your browser to verify it is running."
    )
