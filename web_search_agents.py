import os
from langgraph.prebuilt import create_react_agent
from langchain_ollama.chat_models import ChatOllama
from langchain_tavily import TavilySearch



class WebSearchAgent:
    def __init__(self):
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

        self.web_agent = create_react_agent(
            model=ChatOllama(model="mistral"),
            name="Web Search Bot",
            tools=[
                TavilySearch(
                    max_results=5,
                    topic="general",
                )
            ],
            prompt=(
                "You are a web search expert. Search the web thoroughly to answer user questions with the help of TavilySearch tool."
                "Always provide accurate information based on the documentation."
                "When relevant, include direct links to specific documentation pages that address the user's question."
                "If you're unsure about an answer, acknowledge it and suggest where the user might find more information."
                "Format your responses clearly with headings, bullet points, and code examples when appropriate."
                "Always verify that your answer directly addresses the user's specific question."
            ),
        )
