# JEA RAG Agent - Model Selection Guide

The JEA RAG Agent now supports multiple AI models for generating responses. You can choose between OpenAI's GPT-4o-mini and Google's Gemini Flash 2.0.

## Supported Models

### OpenAI GPT-4o-mini
- **Cost**: Very cost-effective
- **Performance**: Fast and efficient
- **API Key Required**: `OPENAI_API_KEY`
- **Get API Key**: https://platform.openai.com/api-keys

### Google Gemini Flash 2.0
- **Cost**: Free tier available
- **Performance**: Fast and reliable
- **API Key Required**: `GEMINI_API_KEY`
- **Get API Key**: https://aistudio.google.com/app/apikey

## Setup

### 1. Install Required Dependencies

For OpenAI support:
```bash
pip install openai
```

For Gemini support (already included):
```bash
pip install google-generativeai
```

### 2. Configure Environment Variables

Create a `.env` file in your project directory:

```bash
# OpenAI API Key (for GPT-4o-mini)
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini API Key (for Gemini Flash 2.0)
GEMINI_API_KEY=your_gemini_api_key_here
```

**Note**: You only need one API key, but having both gives you more options and fallback capabilities.

## Usage

### Python Code

```python
from rag_agent import RAGAgent

# Use OpenAI GPT-4o-mini
agent = RAGAgent(preferred_model="openai")

# Use Google Gemini Flash 2.0 (default)
agent = RAGAgent(preferred_model="gemini")

# The agent will automatically fallback to available models
agent = RAGAgent()  # Uses gemini by default, falls back to openai if gemini unavailable
```

### Command Line

```bash
# Use OpenAI GPT-4o-mini
python rag_agent.py --model openai "What are JEA's electric rates?"

# Use Google Gemini Flash 2.0
python rag_agent.py --model gemini "What are JEA's electric rates?"

# Interactive mode with model selection
python rag_agent.py --model openai --interactive
```

### Streaming Support

Both models support streaming responses:

```python
# Stream response with chosen model
for chunk in agent.stream_query("What are JEA's payment options?"):
    print(chunk, end='', flush=True)
```

## Model Selection Logic

1. **Preference**: The agent respects your preferred model choice
2. **Availability**: If your preferred model isn't available (missing API key), it falls back to the other
3. **Fallback Order**: 
   - If `preferred_model="openai"` but OpenAI unavailable → Falls back to Gemini
   - If `preferred_model="gemini"` but Gemini unavailable → Falls back to OpenAI
4. **Error Handling**: If no models are available, the agent will provide helpful error messages

## Model Information

You can check which models are available:

```python
agent = RAGAgent(preferred_model="openai")
model_info = agent.security_router.get_model_info()
print(f"Using: {model_info['external_model']}")
print(f"Available: {model_info['models_configured']}")
```

## Features

### All Models Support:
- ✅ Response generation
- ✅ Streaming responses
- ✅ Query ambiguity analysis
- ✅ Response caching
- ✅ Network error handling
- ✅ Confidence scoring

### Model-Specific Features:
- **OpenAI GPT-4o-mini**: Lower latency, excellent instruction following
- **Gemini Flash 2.0**: Free tier, good at reasoning tasks

## Troubleshooting

### "No external models available!"
- Check that you have at least one API key configured in your `.env` file
- Verify your API keys are valid and have sufficient quota

### "OpenAI library not installed"
- Install the OpenAI library: `pip install openai`

### Network/SSL Issues
- The agent has built-in network error handling and will provide helpful error messages
- SSL issues are automatically handled for Gemini API calls

### Model Fallback
- If your preferred model fails, check the logs for fallback messages
- The agent will automatically try the other available model

## Cost Considerations

### OpenAI GPT-4o-mini
- Pay-per-use pricing
- Very cost-effective for most use cases
- Check current pricing at: https://openai.com/pricing

### Google Gemini Flash 2.0
- Free tier available
- Rate limits apply
- Check current limits at: https://ai.google.dev/pricing

## Examples

### Basic Usage
```python
# Initialize with OpenAI
agent = RAGAgent(preferred_model="openai")

# Ask a question
for chunk in agent.stream_query("How do I pay my JEA bill?"):
    print(chunk, end='')

# Get sources
sources = agent.get_last_sources()
for source in sources:
    print(f"Source: {source.title} (Score: {source.similarity_score:.3f})")
```

### Model Comparison
```python
# Test both models
models = ["gemini", "openai"]
query = "What are JEA's electric rates?"

for model in models:
    print(f"\n--- {model.upper()} Response ---")
    agent = RAGAgent(preferred_model=model)
    
    for chunk in agent.stream_query(query):
        print(chunk, end='')
    print("\n")
``` 