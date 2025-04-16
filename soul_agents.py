from agno.agent import Agent
from agno.models.ollama import Ollama
from pipeline import SoulPipeline


class SOULAgent:
    def __init__(self):
        self.soul_rag_pipe = SoulPipeline()

        self.soul_agent = Agent(
            model=Ollama(id="mistral"),
            name="SOUL Bot",
            add_references=True,
            show_tool_calls=True,
            markdown=True,
            retriever=self.soul_rag_pipe.agentic_request_handler,
            role="You have to understand the sentence formation of the provided context and answer the questions related to Ontological (science of being) Leadership similar sentence formation style.",
            description="You are an AI agent who answers questions related to Ontological (science of being) Leadership based ONLY and ONLY on the provided context with the same sentence formation style as that of provided context.",
            instructions="""
            The provided contexts are specifically designed. Understand the sentence formation in the contexts and use the same formation to answer the question based ONLY AND ONLY on the provided context.
            """
        )

    def invoke_agent(self, query):
        return self.soul_agent.run(query)
