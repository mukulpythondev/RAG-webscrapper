from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from dotenv import load_dotenv
import nest_asyncio
import os
import re
import asyncio

nest_asyncio.apply()
load_dotenv()

# Setup embeddings and Gemini client
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# URLs to crawl
urls = [
    "https://chaidocs.vercel.app/youtube/getting-started",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/welcome/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/introduction/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/emmit-crash-course/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/html-tags/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/welcome/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/introduction/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/terminology/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/behind-the-scenes/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/branches/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/diff-stash-tags/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/managing-history/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/github/"
]

def get_collection_name(url):
    match = re.search(r"youtube/([^/]+)", url)
    return match.group(1).replace("-", "_") if match else "default_collection"

# Preprocess and load documents into Qdrant
async def prepare_collections():
    all_collections = set()

    for url in urls:
        loader = WebBaseLoader([url])
        loader.requests_per_second = 1
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        collection_name = get_collection_name(url)
        all_collections.add(collection_name)

        try:
            QdrantVectorStore.from_documents(
                documents=split_docs,
                url="localhost:6333",
                collection_name=collection_name,
                embedding=embeddings,
                force_recreate=False
            )
            print(f"Collection '{collection_name}' created or already exists.")
        except Exception as e:
            print(f"Error creating/updating collection '{collection_name}': {e}")
    
    return all_collections

# Ask Gemini to choose the most relevant collection
def choose_best_collection(query, collections):
    prompt = f"""
You are an intelligent assistant. Choose the most relevant collection name based on the user query.

User Query: "{query}"

Available Collections: {', '.join(collections)}

Return ONLY the best matching collection name (no explanation).
"""
    response = client.models.generate_content(contents=prompt, model="gemini-2.0-flash")
    return response.text.strip().lower().replace("-", "_")

# Generate answer using Gemini with context
def answer_with_gemini(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are a helpful assistant. Answer the following question based on the context.

Context:
{context}

Question: {query}
"""
    response = client.models.generate_content(contents=prompt, model="gemini-2.0-flash")
    return response.text.strip()

# Main chatbot loop
async def run_chatbot():
    all_collections = await prepare_collections()

    print("\nAsk me anything (type 'exit' to quit):\n")
    while True:
        user_query = input("🟢 You: ")
        if user_query.lower() == "exit":
            break

        selected_collection = choose_best_collection(user_query, list(all_collections))
        print(f"📁 Gemini selected: {selected_collection}")

        try:
            retriever = QdrantVectorStore.from_existing_collection(
                url="localhost:6333",
                collection_name=selected_collection,
                embedding=embeddings
            ).as_retriever()

            docs = retriever.get_relevant_documents(user_query)
            answer = answer_with_gemini(user_query, docs)

            print(f"\n🤖 Gemini: {answer}\n")
        except Exception as e:
            print(f"Error: {e}")

# Run the chatbot
asyncio.run(run_chatbot())
