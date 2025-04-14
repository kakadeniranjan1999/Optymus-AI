from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


class SoulDB:
    def __init__(self, collection_name, path, save_db, embed_model, provider):
        self.db = None
        self._collection_name = collection_name
        self._path = path
        self._save_db = save_db
        self._embedding_function = self.get_embedding_model(model=embed_model, provider=provider)

    def get_embedding_model(self, model, provider):
        if provider == "ollama":
            return OllamaEmbeddings(model=model)

    def load_documents(self, documents):
        if self._save_db:
            self.db = Chroma.from_documents(
                documents=documents,
                collection_name=self._collection_name,
                persist_directory=self._path,
                embedding=self._embedding_function
            )
        else:
            self.db = Chroma.from_documents(
                documents=documents,
                collection_name=self._collection_name,
                embedding=self._embedding_function
            )

    def get_retriever(self):
        return self.db.as_retriever()
