import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="Source URL")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="The sources that the agent responded to"
    )


tavily = TavilyClient()


@tool
def search(query: str) -> dict:
    """
    Tool that searches over internet
    Args:
        query : The query to search for
    Returns: the search results
    str
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    # result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo")})
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job posting for ai engineer using langchain in the bay area on Linkedin and list their details"
            )
        }
    )
    print(result["structured_response"].answer)


if __name__ == "__main__":
    main()
