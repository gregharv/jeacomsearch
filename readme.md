# JEA Agentic RAG System

## Overview

This project implements a comprehensive Retrieval Augmented Generation (RAG) system for JEA with multi-level security access. The system crawls websites, processes documents, stores data in SQLite, generates embeddings, and provides intelligent query responses through an agentic framework.

## Architecture

### Data Collection Pipeline
1. **Web Crawler**: Crawls JEA website and other external sources using Playwright for JavaScript-heavy sites
2. **Document Processor**: Extracts text from PDFs and other document formats
3. **Database Storage**: Stores all content in SQLite with metadata and processing status
4. **Embedding Generation**: Creates vector embeddings for semantic search
5. **RAG Agent**: Provides intelligent responses using retrieved context

### Security Levels

The system implements three distinct security levels with progressively broader data access:

#### 🌐 External Level (Public)
- **Data Sources:**
  - Company website content
  - Public PDFs and documents
  - External knowledge bases
- **Access:** Available to general users and external stakeholders
- **Use Cases:** General company information, public policies, customer service

#### 🏢 Internal Level (Employee)
- **Data Sources:**
  - All External level data
  - Internal PDFs and documents
  - Internal knowledge bases
  - SQL query generation for internal databases
  - Employee handbooks and procedures
- **Access:** Available to authenticated employees
- **Use Cases:** Internal operations, employee queries, departmental information

#### 🔒 Sensitive Level (Privileged) - Not planned for implementation
- **Data Sources:**
  - All External and Internal level data
  - Sensitive documents and reports
  - Restricted database tables
  - Real-time operational data
  - Executive and financial information
- **Access:** Available to authorized personnel only
- **Use Cases:** Executive decisions, sensitive operations, confidential analysis

## Technical Stack

### Core Components
- **Web Crawler**: Python with Playwright for JavaScript execution
- **Document Processing**: PyPDF2, pdfplumber for PDF extraction
- **Database**: SQLite for content storage and metadata
- **Embeddings**: Sentence Transformers for vector generation
- **LLM Agents**: 
  - External Level: Google Gemini Flash 2.5
  - Internal Level: TBD (under evaluation)
  - Sensitive Level: Self-hosted Llama model
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

**crawl_log**: Detailed logging for debugging and monitoring
- Records crawling events and errors
- Provides audit trail for data collection

## Process Flow

### 1. Data Collection
The system begins by crawling configured data sources:
- **Website Crawling**: Systematically crawls JEA website and external sources, implementing polite crawling practices
- **PDF Processing**: Extracts text from PDF documents using specialized libraries
- **Content Deduplication**: Uses content hashing to detect and handle duplicate content
- **Metadata Extraction**: Captures titles, URLs, modification dates, and other relevant metadata

### 2. Data Storage
All collected data is stored in SQLite database:
- **Structured Storage**: Organized tables for sources, documents, and logs
- **Change Detection**: Content hashing enables efficient incremental updates
- **Status Tracking**: Processing status fields coordinate the pipeline
- **Security Classification**: Data is tagged with appropriate security levels

### 3. Embedding Generation
Vector embeddings are created for semantic search:
- **Text Chunking**: Documents are split into manageable segments
- **Vector Creation**: Embeddings generated using Sentence Transformers models
- **Index Building**: Efficient vector search index for fast retrieval
- **Metadata Linking**: Embeddings maintain references to source documents

### 4. Agentic RAG System
The intelligent query system provides contextual responses with security-aware LLM routing:
- **Query Processing**: Natural language queries are analyzed and understood
- **Context Retrieval**: Relevant documents retrieved using semantic search with Sentence Transformers
- **Response Generation**: AI agent synthesizes responses using retrieved context
  - **External Level**: Google Gemini Flash 2.5 for public-facing queries
  - **Internal Level**: LLM selection under evaluation for employee queries
  - **Sensitive Level**: Self-hosted Llama model for maximum data security
- **Source Citation**: Responses include references to source materials
- **Security Filtering**: Results filtered based on user access level

## Security Implementation

### Access Control
- **User Authentication**: Secure login system with role-based permissions
- **Data Filtering**: Query results filtered by user security clearance
- **LLM Routing**: Queries routed to appropriate LLM based on security level
- **Data Isolation**: Sensitive data never leaves internal infrastructure
- **Audit Logging**: All access attempts and queries are logged
- **Encryption**: Sensitive data encrypted at rest and in transit

### Data Classification
- **Automatic Tagging**: Content automatically classified during ingestion
- **Manual Override**: Administrative controls for security level adjustment
- **Inheritance Rules**: Documents inherit security levels from their sources
- **LLM Selection**: Security level determines which LLM processes the query
- **Regular Review**: Periodic security classification audits

## Installation and Setup

### Prerequisites
- Python 3.8+
- SQLite 3
- Required Python packages (see requirements.txt)

### Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure data sources in the database
4. Run initial crawl: `python cli.py start {website_url} --javascript`
5. Generate embeddings: `python embeddings.py` (not implemented yet)
6. Configure Gemini Flash 2.5 API for external level in `config.py` (not implemented yet)
7. Start RAG system: `python rag_agent.py` (not implemented yet)

### Configuration
- **Data Sources**: Configure websites and document directories in sources table
- **Security Levels**: Assign appropriate security classifications
- **Crawling Schedule**: Set up periodic refresh cycles
- **Embedding Models**: Configure Sentence Transformers models for semantic search
- **LLM Endpoints**: 
  - Configure Gemini Flash 2.5 API for external level
  - Set up self-hosted Llama model for sensitive level
  - Define internal level LLM once selected

## Use Cases

### External Users
- General company information lookup
- Public policy and procedure queries
- Customer service support
- Regulatory compliance information

### Internal Employees
- Employee handbook and procedure queries
- Internal documentation search
- Database query generation and execution
- Departmental information retrieval

### Privileged Users
- Executive decision support
- Sensitive operational data analysis
- Financial and confidential information access
- Real-time system monitoring and reporting

## Monitoring and Maintenance

### Automated Processes
- **Scheduled Crawling**: Regular updates to keep data current
- **Health Monitoring**: System status and performance tracking
- **Error Handling**: Robust error recovery and notification
- **Data Validation**: Content quality and integrity checks

### Administrative Tools
- **Source Management**: Add, modify, and remove data sources
- **Security Administration**: Manage user permissions and data classifications
- **Performance Tuning**: Optimize crawling and query performance
- **Backup and Recovery**: Data protection and disaster recovery procedures

## Future Enhancements

- **Multi-modal Support**: Image and video content processing
- **Advanced Analytics**: Usage patterns and content insights
- **API Integration**: External system connectivity
- **Mobile Interface**: Mobile-optimized query interface
- **Real-time Updates**: Live data streaming and processing
- **Internal LLM Selection**: Finalize LLM choice for internal security level
- **Model Performance Optimization**: Fine-tune Sentence Transformers and Llama models for JEA-specific content
