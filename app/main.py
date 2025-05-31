from fastapi import FastAPI

from ml.model import load_model
from pydantic import BaseModel

model = None

class SentimentResponse(BaseModel):
    text: str
    sentiment_label: str
    sentiment: float

app = FastAPI()

@app.get('/')
def index():
    return {'text': 'somthing'}

@app.on_event('startup')
def startup_event():
    global model
    model = load_model()
    
@app.get('/predict')
def predict_model(text: str):
    predict = model(text)
    
    response = SentimentResponse(
        text=text,
        sentiment_label=predict.label,
        sentiment_score=predict.score
    )
    return response