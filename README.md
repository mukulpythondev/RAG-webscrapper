# ChaiDocs AI Assistant

An intelligent chatbot that provides answers based on Chai Documentation using LangChain, OpenAI Embeddings, Qdrant Vector Store, and Google's Gemini AI.

## Features

- Crawls and indexes content from Chai Documentation
- Uses OpenAI embeddings for semantic search
- Stores document vectors in Qdrant
- Intelligent collection selection using Gemini AI
- Context-aware responses using RAG (Retrieval Augmented Generation)

## Prerequisites

- Python 3.8+
- Qdrant running locally or remotely
- OpenAI API key
- Google Gemini API key



## Installation

1. Clone the repository:
```bash
git clone https://github.com/mukulpythondev/RAG-webscrapper
cd chaidocs
```

2. Install UV (Fast Python Package Installer):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and activate virtual environment with UV:
```bash
uv venv .venv
source .venv/bin/activate  # On Linux/Mac
```

4. Install dependencies using UV (much faster than pip):
```bash
uv pip install -r requirements.txt
```

5. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

> **Note**: UV is significantly faster than pip for package installation and dependency resolution. It's written in Rust and provides better performance.


## Usage

1. Start Qdrant server:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Run the chatbot:
```bash
python main.py
```

3. Ask questions about Chai documentation. Type 'exit' to quit.

## How it Works

1. **Document Processing**:
   - Crawls specified URLs from Chai documentation
   - Splits documents into chunks
   - Creates embeddings using OpenAI

2. **Vector Storage**:
   - Stores document vectors in Qdrant
   - Organizes content into topic-based collections

3. **Query Processing**:
   - Uses Gemini to select relevant collection
   - Retrieves similar documents using vector search
   - Generates context-aware responses

## Project Structure

```
chaidocs/
├── main.py           # Main application code
├── .env             # Environment variables
├── .gitignore       # Git ignore file
├── README.md        # Project documentation
└── requirements.txt # Project dependencies
```

## Dependencies

- langchain
- openai
- google-generative-ai
- qdrant-client
- python-dotenv
- nest-asyncio

