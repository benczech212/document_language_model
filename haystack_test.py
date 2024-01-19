import haystack_test
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader, DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline

def extract_text_from_files(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            from pdfminer.high_level import extract_text
            text = extract_text(file_path)
        elif file_path.endswith((".docx", ".doc")):
            import docx2txt
            text = docx2txt.process(file_path)
        # Add similar extraction for other file types
        documents.append({"content": text, "meta": {"name": file_path}})
    return documents
