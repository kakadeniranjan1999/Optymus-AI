from pipeline import SoulPipeline


chatbot = SoulPipeline()

def chat(msg, model):
    return chatbot.rag_request_handler(msg, model)
