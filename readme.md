# JEA Agentic RAG System

## Overview

This project implements a comprehensive Retrieval Augmented Generation (RAG) system for JEA with multi-level security access. The system crawls websites, processes documents, stores data in SQLite, generates embeddings, and provides intelligent query responses through an agentic framework with a user-friendly Streamlit interface.

## Architecture

### Data Collection Pipeline
1. **Web Crawler**: Crawls JEA website and other external sources using Playwright for JavaScript-heavy sites
2. **Document Processor**: Extracts text from PDFs and other document formats
3. **Database Storage**: Stores all content in SQLite with metadata and processing status
4. **Embedding Generation**: Creates vector embeddings for semantic search using Sentence Transformers
5. **RAG Agent**: Provides intelligent responses using retrieved context with LLM-powered analysis

### Security Levels

The system implements three distinct security levels with progressively broader data access:

#### 🌐 External Level (Public) - ✅ IMPLEMENTED
- **Data Sources:**
  - Company website content
  - Public PDFs and documents
  - External knowledge bases
- **LLM Model:** Google Gemini Flash 2.5 Experimental
- **Access:** Available to general users and external stakeholders
- **Use Cases:** General company information, public policies, customer service

#### 🏢 Internal Level (Employee) - 🔄 PLANNED
- **Data Sources:**
  - All External level data
  - Internal PDFs and documents
  - Internal knowledge bases
  - Employee handbooks and procedures
- **LLM Model:** TBD (under evaluation)
- **Access:** Available to authenticated employees
- **Use Cases:** Internal operations, employee queries, departmental information

#### 🔒 Sensitive Level (Privileged) - 🔄 PLANNED
- **Data Sources:**
  - All External and Internal level data
  - Sensitive documents and reports
  - Restricted database tables
  - Real-time operational data
- **LLM Model:** Self-hosted Llama model (for maximum data security)
- **Access:** Available to authorized personnel only
- **Use Cases:** Executive decisions, sensitive operations, confidential analysis

## Technical Stack

### Core Components
- **Web Crawler**: Python with Playwright for JavaScript execution
- **Document Processing**: PyPDF2, pdfplumber for PDF extraction
- **Database**: SQLite for content storage and metadata
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) with local caching and UNC path support
- **LLM Agents**: 
  - External Level: Google Gemini Flash 2.5 Experimental ✅
  - Internal Level: TBD (under evaluation) 🔄
  - Sensitive Level: Self-hosted Llama model 🔄
- **User Interface**: Streamlit web application with real-time streaming responses ✅
- **Security**: Based on connection from local network or external network

### Database Schema

The system uses SQLite with the following core tables:

**sources**: Manages data source configurations
- Tracks website URLs, PDF directories, and other data sources
- Maintains crawling status and metadata
- Associates sources with security levels

**documents**: Stores processed content
- Contains extracted text from web pages and documents
- Tracks content hashes for change detection
- Maintains processing status for embedding pipeline

**embeddings**: Stores vector embeddings for semantic search
- Links to document chunks for efficient retrieval
- Supports cosine similarity calculations
- Optimized for fast vector operations

**crawl_log**: Detailed logging for debugging and monitoring
- Records crawling events and errors
- Provides audit trail for data collection

**rag_interactions**: Logs user interactions for monitoring
- Query tracking and performance analysis
- Security level and confidence scoring
- User context and response metadata

## Current Implementation Status

### ✅ Fully Implemented Features

#### RAG Agent (rag_agent.py)
- **Multi-Strategy Retrieval**: Enhanced document retrieval with multiple search variants
- **LLM-Powered Query Analysis**: Intelligent query intent detection and ambiguity analysis
- **Security-Aware Routing**: Automatic LLM selection based on security levels
- **Streaming Responses**: Real-time token streaming for better user experience
- **Conversation Context**: Multi-turn conversation support with context memory
- **Source Management**: Comprehensive source tracking and citation
- **Error Handling**: Robust error recovery and graceful degradation
- **Local Model Caching**: Efficient embedding model storage with UNC path support

#### Streamlit Interface (streamlit_app.py)
- **Real-Time Streaming**: Live response generation with typing indicator
- **Conversation History**: Complete chat history with timestamps and confidence scores
- **Search Modes**: High Reasoning (comprehensive) vs Fast Search modes
- **Source Display**: Expandable source citations with relevance scores
- **Performance Metrics**: Conversation statistics and confidence tracking
- **Responsive Design**: Clean, professional UI with JEA branding
- **Error Handling**: User-friendly error messages and recovery

#### Advanced Features
- **Enhanced Retrieval**: Multiple query variants with result combination and re-ranking
- **Query Ambiguity Detection**: LLM-powered analysis to determine when clarification is needed
- **Context-Aware Responses**: Conversation memory for natural follow-up questions
- **Assumption-Based Processing**: Intelligent defaults for common utility queries
- **Performance Optimization**: Configurable search modes for speed vs accuracy trade-offs

### 🔄 In Progress / Planned

#### Security Implementation
- **User Authentication**: Role-based access control system
- **Internal Level LLM**: Selection and integration of internal security level model
- **Sensitive Level Setup**: Self-hosted Llama model deployment
- **Access Control**: User permission and data filtering systems

#### Enhanced Features
- **Multi-modal Support**: Image and video content processing
- **Advanced Analytics**: Usage patterns and content insights
- **API Integration**: RESTful API for external system connectivity
- **Mobile Optimization**: Enhanced mobile interface

## Process Flow

### 1. Data Collection
The system begins by crawling configured data sources:
- **Website Crawling**: Systematically crawls JEA website and external sources
- **PDF Processing**: Extracts text from PDF documents using specialized libraries
- **Content Deduplication**: Uses content hashing to detect and handle duplicate content
- **Metadata Extraction**: Captures titles, URLs, modification dates, and other relevant metadata

### 2. Data Storage and Embedding
All collected data is processed and vectorized:
- **Structured Storage**: Organized tables for sources, documents, and embeddings
- **Vector Generation**: Sentence Transformers create embeddings with local caching
- **Change Detection**: Content hashing enables efficient incremental updates
- **Index Building**: Efficient vector search with cosine similarity

### 3. Intelligent Query Processing
The RAG agent provides sophisticated query handling:

#### Query Analysis
- **Intent Detection**: Determines query type (rates, contact, payment assistance, etc.)
- **Ambiguity Analysis**: LLM-powered detection of unclear queries
- **Search Strategy**: Multiple query variants for comprehensive retrieval
- **Context Integration**: Conversation memory for follow-up questions

#### Document Retrieval
- **Semantic Search**: Vector similarity using Sentence Transformers
- **Multi-Query Approach**: Enhanced retrieval with multiple search variants
- **Result Re-ranking**: Secondary scoring based on original query
- **Quality Filtering**: Relevance thresholds and result optimization

#### Response Generation
- **LLM Selection**: Security-aware model routing (currently Gemini Flash 2.5)
- **Context Building**: Structured prompts with retrieved documents
- **Streaming Output**: Real-time token generation
- **Source Citation**: Automatic reference linking and relevance scoring

### 4. User Interface
Streamlit application provides intuitive access:
- **Real-time Streaming**: Live response generation with visual feedback
- **Conversation Management**: History, context, and follow-up support
- **Source Transparency**: Expandable citations with relevance metrics
- **Performance Control**: High Reasoning vs Fast Search modes

## Installation and Setup

### Prerequisites
- Python 3.8+
- SQLite 3
- Required Python packages (see requirements.txt)
- Google API key for Gemini Flash 2.5

### Quick Start
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd jea-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Create .env file with your API keys
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

4. **Set up data sources**
   ```bash
   # Configure data sources in the database
   python cli.py start {website_url} --javascript
   ```

5. **Generate embeddings**
   ```bash
   # Process documents and create embeddings
   python embeddings.py
   ```

6. **Start the application**
   ```bash
   # Launch Streamlit interface
   streamlit run streamlit_app.py
   ```

### Configuration

#### Embedding Model Setup
The system automatically handles embedding model setup with local caching:
- **Model**: all-MiniLM-L6-v2 (Sentence Transformers)
- **Local Cache**: Configurable cache directory with UNC path support
- **Fallback**: Automatic download if local cache unavailable

#### LLM Configuration
- **External Level**: Google Gemini Flash 2.5 Experimental
  - Requires GEMINI_API_KEY in .env file
  - Automatic SSL verification handling for downloads
- **Internal/Sensitive Levels**: Placeholder for future implementation

#### Data Sources
- **Website Crawling**: Configure URLs in sources table
- **Security Classification**: Assign appropriate security levels
- **Refresh Scheduling**: Set up periodic update cycles

## Use Cases and Examples

### Typical User Interactions

#### Customer Service Queries
```
User: "What are the current electric rates?"
Assistant: Provides current residential electric rate structure with rate components, fuel charges, and links to official rate schedules.
Sources: 3 relevant documents with rate information
```

#### Payment Assistance
```
User: "I'm having trouble paying my bill. What options do I have?"
Assistant: Lists payment plan options, financial assistance programs, budget billing, and contact information for customer assistance.
Sources: Payment assistance program documents and customer service pages
```

#### General Information
```
User: "What are JEA's customer service hours?"
Assistant: Provides current customer service hours, phone numbers, and mentions other department hours if relevant.
Sources: Contact information and customer service pages
```

### Advanced Features in Action

#### Conversation Context
```
User: "What are electric rates?"
Assistant: [Provides detailed rate information]
User: "How do those compare to last year?"
Assistant: [Uses conversation context to understand "those" refers to electric rates]
```

#### High Reasoning Mode
- **Multiple Query Variants**: "electric rates" → ["electric rates", "electricity pricing", "rate schedule"]
- **Enhanced Filtering**: Prioritizes documents with pricing information
- **Result Combination**: Merges and re-ranks results from multiple searches

#### Ambiguity Handling
- **Intelligent Defaults**: Assumes "rates" means "residential rates" unless specified
- **Context Clues**: Uses available information to provide helpful answers
- **Clarification Requests**: Only asks for clarification when truly necessary

## Monitoring and Performance

### Real-Time Metrics
- **Response Times**: Streaming performance and total response time
- **Confidence Scores**: LLM confidence in responses (0.0-1.0)
- **Source Quality**: Relevance scores and document match quality
- **Search Performance**: High Reasoning vs Fast Search mode comparison

### Conversation Analytics
- **Session Tracking**: Multi-turn conversation management
- **Context Effectiveness**: Conversation memory impact on responses
- **User Satisfaction**: Confidence trends and source utilization

### System Health
- **Model Performance**: Embedding and LLM response quality
- **Database Efficiency**: Query performance and storage optimization
- **Error Rates**: Exception handling and recovery success

## Future Enhancements

### Short Term (Next Quarter)
- **User Authentication**: Role-based access control implementation
- **Internal LLM Integration**: Selection and deployment of internal security level model
- **Performance Optimization**: Query caching and response time improvements
- **Mobile Interface**: Enhanced mobile responsiveness

### Medium Term (6 Months)
- **Sensitive Level Implementation**: Self-hosted Llama model deployment
- **Advanced Analytics**: Usage patterns and content insights dashboard
- **API Development**: RESTful API for external system integration
- **Multi-modal Support**: Image and document content processing

### Long Term (1 Year)
- **Real-time Data Integration**: Live operational data streaming
- **Predictive Analytics**: Usage pattern prediction and content recommendations
- **Voice Interface**: Speech-to-text query support
- **Advanced Security**: Enhanced access controls and audit systems

## Technical Details

### Embedding Pipeline
- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Chunking Strategy**: Document-level embeddings with metadata preservation
- **Similarity Calculation**: Cosine similarity with normalized vectors
- **Performance**: ~1000 documents/minute processing rate

### LLM Integration
- **External Level**: Google Gemini Flash 2.5 Experimental
  - Streaming support for real-time responses
  - Context window: 2M tokens
  - Response quality optimized for customer service
- **Prompt Engineering**: Structured prompts with source context and conversation history
- **Error Handling**: Graceful degradation and fallback responses

### Security Architecture
- **Data Classification**: Automatic security level assignment
- **Access Control**: User-based filtering (planned)
- **Audit Logging**: Comprehensive interaction tracking
- **Data Isolation**: Security level-appropriate model routing