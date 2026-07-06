import streamlit as st
import requests
import extra_streamlit_components as stx
import time

FASTAPI_URL = "http://127.0.0.1:8000"

st.title("Streamlit + JWT Persistent Auth")

# 1. Setup layout-safe session tracking keys
if "component_ready" not in st.session_state:
    st.session_state["component_ready"] = False
if "token" not in st.session_state:
    st.session_state["token"] = None

# 2. Instantiate Cookie Manager
cookie_manager = stx.CookieManager(key="persistent_cookie_auth")

# 3. CRITICAL: Pause execution to let the iframe component sync with the browser
if not st.session_state["component_ready"]:
    with st.spinner("Syncing security credentials..."):
        time.sleep(0.5)  # Wait for iframe component lifecycle to attach
    st.session_state["component_ready"] = True
    st.rerun()
    st.stop()

# 4. Read cookies safely now that the synchronization window has closed
all_cookies = cookie_manager.get_all()

# Extract token if the dictionary exists
saved_token = None
if isinstance(all_cookies, dict):
    saved_token = all_cookies.get("auth_token")

# Update state if the browser holds a valid token
if saved_token and st.session_state["token"] != saved_token:
    st.session_state["token"] = saved_token
    st.rerun()
    st.stop()

# 5. Login Form View
if not st.session_state["token"]:
    st.subheader("Login to your Account")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Log In"):
        response = requests.post(
            f"{FASTAPI_URL}/login",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            token = response.json().get("access_token")

            # Persist globally and locally
            st.session_state["token"] = token
            cookie_manager.set("auth_token", token, max_age=86400)

            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# 6. Authenticated Dashboard View
else:
    st.subheader("Secure Dashboard")
    st.success("Authenticated State Preserved!")

    if st.button("Fetch Secure Data"):
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        res = requests.get(f"{FASTAPI_URL}/protected-data", headers=headers)

        if res.status_code == 200:
            st.json(res.json())
        elif res.status_code == 401:
            st.error("Session expired or invalid token. Please log in again.")
            cookie_manager.delete("auth_token")
            st.session_state["token"] = None
            st.rerun()
        else:
            st.error("An error occurred.")

    if st.button("Log Out"):
        cookie_manager.delete("auth_token")
        st.session_state["token"] = None
        # Reset ready flag to allow fresh syncing upon next access
        st.session_state["component_ready"] = False
        st.rerun()
