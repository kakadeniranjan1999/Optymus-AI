from pipeline import SoulPipeline


chatbot = SoulPipeline()

def chat(msg, model):
    return chatbot.chat(msg, model)
