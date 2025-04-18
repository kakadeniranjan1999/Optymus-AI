from optymus_team import OptymusTeam
import time
from dotenv import load_dotenv

load_dotenv()
team_optymus = OptymusTeam()

def chat(msg, model):
    query = {
        "messages": [
            {
                "role": "user",
                "content": msg
            }
        ]
    }

    response = team_optymus.invoke_team(query)

    # print(response)

    for chunk in response["messages"][1].content:
        yield chunk
        time.sleep(0.005)
