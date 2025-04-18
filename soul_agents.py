from langgraph.prebuilt import create_react_agent
from langchain_ollama.chat_models import ChatOllama
from pipeline import SoulPipeline


class SOULAgent:
    def __init__(self):
        self.soul_rag_pipe = SoulPipeline()

        self.soul_agent = create_react_agent(
            model=ChatOllama(model="mistral"),
            name="SOUL Bot",
            tools=[self.invoke_agent],
            prompt="""
            The provided contexts are specifically designed. Understand the sentence formation in the contexts and use the same formation to answer the question based ONLY AND ONLY on the provided context.
            """
        )

    def invoke_agent(self, query):
        """
        An AI agent who answers questions related to Ontological (science of being) Leadership based ONLY and ONLY on the provided context with the same sentence formation style as that of provided context.
        """
        return self.soul_rag_pipe.agentic_request_handler(query)
