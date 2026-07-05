## Project architecture - RAG+VectorDB+LLM
# The agent ui calls rest apis for the agent
# Run the server 
```
 uvicorn agent:app --reload --port 8000
```
# Run the chat bot ui app
```
streamlit run agent-ui.py 
```