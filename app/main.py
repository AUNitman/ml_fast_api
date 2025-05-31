from fastapi import FastAPI

from ml.model import load_model
from pydantic import BaseModel

model = None

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment_label: str
    sentiment: float
    
class ModelStatus(BaseModel):
    status: str
    model_name: str
    error: str = None

app = FastAPI()

@app.get('/status', response_model=ModelStatus)
def index():
    return {'text': 'somthing'}

@app.on_event('startup')
def startup_event():
    global model
    model = load_model()
    
@app.get('/analyze', response_model=SentimentResponse)
def predict_model(text: str):
    predict = model(text)
    
    response = SentimentResponse(
        text=text,
        sentiment_label=predict.label,
        sentiment_score=predict.score
    )
    return response