import os
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools


class WebSearchAgent:
    def __init__(self):
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

        self.web_agent = Agent(
            model=Ollama(id="mistral"),
            name="Web Search Bot",
            show_tool_calls=True,
            markdown=True,
            tools=[TavilyTools(), DuckDuckGoTools()],
            role="You have to search web for the given query and provide a meaningful response.",
            description="You are an AI agent who searches the web for the given query and provides a meaningful response for the same.",
            instructions=[
                "You are a web search expert. Search the web thoroughly to answer user questions.",
                "Always provide accurate information based on the documentation.",
                "When relevant, include direct links to specific documentation pages that address the user's question.",
                "If you're unsure about an answer, acknowledge it and suggest where the user might find more information.",
                "Format your responses clearly with headings, bullet points, and code examples when appropriate.",
                "Always verify that your answer directly addresses the user's specific question.",
                "If you cannot find the answer in the documentation knowledge base, use the TavilyTools or DuckDuckGoTools to search the web for relevant information to answer the user's question.",
            ],
        )

    def invoke_agent(self, query):
        return self.web_agent.run(query)
