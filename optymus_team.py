from soul_agents import SOULAgent
from web_search_agents import WebSearchAgent
from agno.team import Team
from agno.models.ollama import Ollama



class OptymusTeam:
    def __init__(self):
        self.soul_agent = SOULAgent().soul_agent
        self.web_agent = WebSearchAgent().web_agent

        self.optymus_team = Team(
            name="AutoBots",
            mode="route",
            model=Ollama(id="mistral"),
            enable_team_history=True,
            members=[self.soul_agent, self.web_agent],
            show_tool_calls=True,
            markdown=True,
            debug_mode=True,
            show_members_responses=True,
            instructions=[
                "You are the lead agent responsible for classifying and routing user queries.",
                "Carefully analyze each user message and determine if it is: a question that needs leadership expertise or web search expertise.",
                "For leadership questions, route to the soul_agent",
                "For other questions, route to the web_agent",
                "Always provide a clear explanation of why you're routing the inquiry to a specific agent.",
                "After receiving a response from the appropriate agent, relay that information back to the user in a professional and helpful manner.",
                "Ensure a seamless experience for the user by maintaining context throughout the conversation.",
            ],
        )

    def invoke_team(self, query):
        return self.optymus_team.run(query)
