from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

import uuid

from ml.model import load_model

model = None

result_storage: Dict[str, dict] = {}

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    
class ModelStatus(BaseModel):
    status: str
    model_name: str

app = FastAPI(
    title='Classifiret Text'
)

@app.on_event('startup')
def startup_event():
    global model
    try:
        model = load_model()
    except Exception as e:
        raise RuntimeError(f'Error to load model: {str(e)}')
    
@app.post('/submit', response_model=SentimentResponse)
async def predict_model(text: str):
    
    try:
        result = model(text)
        
        response = SentimentResponse(
            text=text,
            sentiment=result.label,
            confidence=result.score
        )
        result_id = str(uuid.uuid4())
        result_storage[result_id] = {
            'text': text,
            'status': 'completed',
            'name': result.name
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=0, detail=f"Error in predict model: {str(e)}")

@app.get('/status', response_model=ModelStatus)
async def model_status():
    status = 'ready' if model is not None else 'not loaded'
    return ModelStatus(
        status=status,
        model_name=getattr(model, 'name', None),
    )
