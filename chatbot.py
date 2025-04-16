from optymus_team import OptymusTeam
import time
from dotenv import load_dotenv

load_dotenv()
team_optymus = OptymusTeam()

def chat(msg, model):
    for chunk in team_optymus.invoke_team(msg).content:
        yield chunk
        time.sleep(0.005)
