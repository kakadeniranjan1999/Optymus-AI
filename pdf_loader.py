from langchain_community.document_loaders import PyPDFDirectoryLoader


class PDFLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.loader = PyPDFDirectoryLoader(
            path = self.folder_path,
            glob = "*.pdf",
            mode = "page",
            extraction_mode = "plain",
        )

    def load_documents(self):
        return self.loader.load()