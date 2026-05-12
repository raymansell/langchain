import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader(
        "/Users/raymondmansell/Documents/langchain-course/mediumblog1.txt",
        encoding="UTF-8"
    )
    document = loader.load()

    print("Splitting document(s) into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(document)
    print(f"created {len(text_chunks)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting chunks as vectors (via the OpenAI text-embedding-ada-002 embedding model)...")
    PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=os.environ["INDEX_NAME"])

    print("finish")