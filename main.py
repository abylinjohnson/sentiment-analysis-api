from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
import time
from typing import Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment using DistilBERT model",
    version="1.0.0"
)

# Initialize the model globally
model = None

class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float
    processing_time_seconds: float

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model
    model = pipeline("sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english")

@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Sentiment Analysis API",
        "usage": "Send POST request to /analyze with JSON body containing 'text' field"
    }

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput) -> Dict[str, Any]:
    """
    Analyze sentiment of input text using DistilBERT.
    
    Args:
        input_data: TextInput object containing text to analyze
    
    Returns:
        Dictionary containing sentiment analysis results
    """
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    # Ensure model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # Record start time
        start_time = time.time()
        
        # Get prediction
        result = model(input_data.text)[0]
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 3)
        
        return {
            "text": input_data.text,
            "label": result["label"],
            "score": round(result["score"], 3),
            "processing_time_seconds": processing_time
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)