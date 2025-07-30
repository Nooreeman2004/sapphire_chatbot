# Sapphire RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) pipeline for Sapphire, a Pakistani fashion brand. This system provides intelligent customer service through a chatbot that can answer questions about products, policies, and brand information.

## 🌟 Features

- **Advanced RAG Pipeline**: Semantic chunking, contextual retrieval, and response generation
- **Personalized Experience**: User name input and personalized greetings
- **Quick Action Buttons**: 6 predefined quick actions for common customer queries
- **Professional Architecture**: Modular design with clear separation of concerns
- **Multiple Interfaces**: FastAPI backend with enhanced Streamlit frontend
- **Intelligent Chunking**: Semantic product chunking and Q&A processing
- **Vector Search**: FAISS-powered similarity search with sentence transformers
- **LLM Integration**: Groq Cloud API for fast and accurate response generation
- **Conversation Management**: Session-based chat history and context awareness
- **Enhanced UI**: Modern, responsive design with improved visibility and user experience
- **Graceful Error Handling**: Fallback responses when API is unavailable
- **Scalable Design**: Professional development structure ready for production

## 🏗️ Architecture

```
sapphire_chatbot/
├── config/
│   └── config.yaml                # Configuration settings
├── data/
│   ├── *.json                     # Source data files
│   └── processed_chunks/          # Processed document chunks
├── src/
│   ├── data_processing/           # Data chunking and preprocessing
│   ├── embeddings/                # Vector embeddings and storage
│   ├── retrieval/                 # Advanced retrieval with reranking
│   ├── generation/                # LLM response generation
│   ├── api/                       # FastAPI routes and models
│   └── utils/                     # Logging and helper utilities
├── vector_store/                  # FAISS vector database
├── logs/                          # Application logs
├── main.py                        # FastAPI entry point
├── app.py                         # Streamlit frontend
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/))

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd sapphire_chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
   ```

4. **Process data and build vector store:**
   ```bash
   python -c "
   from src.data_processing.chunking import process_all_data
   process_all_data()
   "
   ```

5. **Start the FastAPI backend:**
   ```bash
   python main.py
   ```

6. **Launch the Streamlit frontend (in a new terminal):**
   ```bash
   streamlit run app.py
   ```

7. **Access the application:**
   - Frontend: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

## 📊 Data Processing

The system processes various types of data from Sapphire's website:

### Supported Data Types

- **Products**: Comprehensive product information including details, pricing, and availability
- **FAQs**: Frequently asked questions and answers
- **Policies**: Return, exchange, and shipping policies
- **General Content**: Brand information and other website content

### Chunking Strategy

- **Semantic Product Chunking**: Products are split into logical chunks (overview, details, care instructions, availability)
- **Q&A Chunking**: FAQ pairs are processed as question-answer chunks
- **Content Chunking**: Web content is chunked with overlap for context preservation

## 🔧 Configuration

The system is highly configurable through `config/config.yaml`:

```yaml
# Model Configuration
model:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  llm:
    provider: "groq"
    model_name: "mixtral-8x7b-32768"
    temperature: 0.1

# Vector Store Configuration
vector_store:
  type: "langchain"
  backend: "faiss"
  search:
    default_k: 5

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
```

## 🔌 API Endpoints

### Chat Endpoints

- `POST /chat` - Basic chat with retrieval
- `POST /chat/advanced` - Advanced chat with full RAG pipeline
- `GET /conversations/{session_id}/history` - Get conversation history
- `DELETE /conversations/{session_id}` - Clear conversation

### Search and Utility

- `POST /search` - Search knowledge base
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /config` - Update configuration

### Example API Usage

```python
import requests

# Send a chat message
response = requests.post("http://localhost:8000/chat", json={
    "message": "What's your return policy?",
    "session_id": "user123",
    "include_sources": True
})

print(response.json())
```

## 🎨 Frontend Features

The Streamlit frontend provides:

- **Clean Interface**: Simple chat interface focused on user experience
- **Real-time Chat**: Instant responses with typing indicators
- **Session Management**: Persistent conversations with history
- **Source Display**: Optional source information for transparency
- **Responsive Design**: Works on desktop and mobile devices

## 🧠 RAG Pipeline Details

### 1. Data Processing
- JSON data parsing and cleaning
- Semantic chunking based on content type
- Metadata extraction and enrichment

### 2. Embedding Generation
- Sentence Transformers (all-MiniLM-L6-v2)
- Batch processing for efficiency
- Vector storage in FAISS

### 3. Retrieval
- Similarity search with configurable k
- History-aware query rewriting
- Contextual compression (optional)

### 4. Generation
- Groq Cloud API integration
- Contextual prompt engineering
- Response post-processing

## 🔍 Advanced Features

### History-Aware Retrieval
The system maintains conversation context and rewrites queries based on chat history for better retrieval accuracy.

### Reranking (Configurable)
Optional reranking of retrieved documents using LLM-based scoring for improved relevance.

### Session Management
- Automatic session creation
- Conversation history tracking
- Session timeout and cleanup

### Error Handling
- Comprehensive error handling and logging
- Graceful degradation for API failures
- User-friendly error messages

## 📝 Logging

The system includes comprehensive logging:

- **File Rotation**: Automatic log file rotation
- **Multiple Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Structured Logging**: Consistent format across components
- **Performance Tracking**: Response times and error rates

## 🧪 Testing

### Manual Testing

1. **Start the system** following the Quick Start guide
2. **Test basic functionality**:
   - Ask about products: "Show me summer dresses"
   - Ask about policies: "What's your return policy?"
   - Ask about sizing: "What sizes do you have?"

3. **Test advanced features**:
   - Multi-turn conversations
   - Context awareness
   - Source information

### API Testing

Use the interactive API documentation at `http://localhost:8000/docs` to test endpoints directly.

## 🚀 Deployment

### Production Considerations

1. **Environment Variables**:
   ```bash
   export GROQ_API_KEY="your-production-key"
   export HOST="0.0.0.0"
   export PORT="8000"
   export LOG_LEVEL="INFO"
   ```

2. **Security**:
   - Enable API key authentication
   - Configure rate limiting
   - Set up HTTPS

3. **Scaling**:
   - Use multiple worker processes
   - Implement caching
   - Monitor performance

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

## 🛠️ Development

### Project Structure

- **Modular Design**: Clear separation between data processing, retrieval, and generation
- **Type Hints**: Full type annotation for better code quality
- **Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Robust error handling throughout

### Adding New Features

1. **New Data Sources**: Add processors in `src/data_processing/`
2. **Custom Retrievers**: Extend `src/retrieval/retriever.py`
3. **API Endpoints**: Add routes in `src/api/routes.py`
4. **Frontend Components**: Modify `app.py`

## 📋 Requirements

### Core Dependencies

- **FastAPI**: Web framework for the API
- **Streamlit**: Frontend framework
- **LangChain**: RAG pipeline framework
- **Sentence Transformers**: Embedding generation
- **FAISS**: Vector similarity search
- **Groq**: LLM API integration

### System Requirements

- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ for vector store and models
- **Network**: Internet connection for LLM API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

1. **API Connection Errors**: Ensure Groq API key is set correctly
2. **Vector Store Loading**: Verify vector store was built successfully
3. **Memory Issues**: Reduce batch size in configuration
4. **Port Conflicts**: Change port in configuration

### Getting Help

- Check the logs in `./logs/` for detailed error information
- Use the API documentation at `/docs` for endpoint details
- Review the configuration in `config/config.yaml`

## 🔄 Updates and Maintenance

### Regular Maintenance

1. **Update Dependencies**: Regularly update Python packages
2. **Monitor Logs**: Check for errors and performance issues
3. **Backup Data**: Backup vector store and configuration
4. **Performance Tuning**: Adjust configuration based on usage

### Version History

- **v1.0.0**: Initial release with full RAG pipeline
- Features: Semantic chunking, FAISS vector store, Groq integration, Streamlit frontend

---

**Built with ❤️ for Sapphire Fashion Brand**

