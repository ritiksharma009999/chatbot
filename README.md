# FastAPI Semantic Search Chatbot

A powerful semantic search API built with FastAPI, Qdrant vector database, and LlamaIndex. This application provides advanced document search capabilities using state-of-the-art language models and vector search technology.

## Features

- **Semantic Search**: Utilizes embeddings for meaningful search results
- **Vector Database**: Powered by Qdrant for efficient vector similarity search
- **REST API**: Built with FastAPI for high performance
- **AWS Integration**: Uses AWS Bedrock for advanced language model capabilities
- **Hybrid Search**: Combines vector and sparse search for better results
- **Docker Support**: Fully containerized application

## Tech Stack

- FastAPI
- Qdrant Vector Database
- LlamaIndex
- AWS Bedrock
- Docker & Docker Compose
- Python 3.12

## Project Structure

```
chatbot/
├── src/
│   ├── __init__.py
│   ├── bot.py           # FastAPI application
│   └── preprocess.py    # Data indexing and preprocessing
├── data/                # Document storage
├── Dockerfile          
├── docker-compose.yml   
├── requirements.txt     
└── .env                # Environment variables (not tracked)
```

## Prerequisites

- Docker and Docker Compose
- AWS Account with Bedrock access
- Qdrant instance running
- Python 3.12+ (for local development)

## Environment Variables

Create a `.env` file with:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-northeast-1
QDRANT_HOST=your_qdrant_host
QDRANT_PORT=6333
```

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/ritiksharma009999/chatbot.git
cd chatbot
```

2. Build and start the Docker containers:
```bash
docker-compose up --build
```

3. The API will be available at:
```
http://localhost:8000
```

## API Documentation

Access the interactive API documentation at:
```
http://localhost:8000/docs
```

### API Endpoints

#### POST /query
Performs semantic search on indexed documents.

Request body:
```json
{
  "query": "Your search query",
  "similarity_top_k": 10,
  "sparse_top_k": 12
}
```

Response:
```json
{
  "answer": "Generated response",
  "sources": [
    {
      "node_id": "...",
      "content": "...",
      "colbert_score": 0.95,
      "retrieval_score": 0.85
    }
  ]
}
```

## Development

For local development:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn src.bot:app --reload --host 0.0.0.0 --port 8000
```

## Docker Commands

Common Docker commands for managing the application:

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

## Testing

Test the API using curl:

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{
  "query": "What is Ubuntu?",
  "similarity_top_k": 10,
  "sparse_top_k": 12
}'
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository.
