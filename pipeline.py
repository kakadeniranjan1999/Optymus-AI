from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pdf_loader import PDFLoader
from text_splitter import TextSplitter
from database import SoulDB


class SoulPipeline:
    def __init__(self):
        self.pdf_loader = PDFLoader(folder_path="./documents")

        self.text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)

        self.vector_db = SoulDB(
            collection_name="soul-db",
            path="./database",
            save_db=False,
            embed_model="nomic-embed-text",
            provider="ollama"
        )

        self.chain = None
        self.model = None
        self.llm = None

        self.setup_db()

        self.retriever = self.vector_db.get_retriever()

        self.prompt_template = """The following provided contexts are specifically designed. Understand the sentence formation in the following contexts and use the same to answer the question based ONLY AND ONLY on the following context. :
                {context}
                Question: {question}
                """

        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

    def setup_db(self):
        documents = self.pdf_loader.load_documents()

        chunks = self.text_splitter.split_text(documents)

        self.vector_db.load_documents(documents=chunks)

    def load_model(self):
        self.llm = OllamaLLM(model=self.model)

    def chat(self, msg, model):
        if model != self.model:
            self.model = model
            self.load_model()

        context = self.retriever.invoke(msg)

        message = self.prompt.invoke(
            {
                "question": msg,
                "context": context
            }
        )

        return self.llm.stream(message)
