from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from ml.model import load_model

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    
class ModelStatus(BaseModel):
    status: str = 'completed'
    result_id: str

model = None

response_storage: Dict[str, SentimentResponse] = {}
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
    
@app.post('/submit', response_model=ModelStatus, summary='Узнать тональность введенного  текста')
async def predict_model(text: str):
    '''
    **text**: текст, тональность которого будет определяться
    
    результатом будет id,  по которому можно найти результат запроса
    '''
    try:
        result = model(text)
        
        response = SentimentResponse(
            text=text,
            sentiment=result.label,
            confidence=result.score
        )
        
        result_id = str(len(response_storage) + 1)
        response_storage[result_id] = response
        
        return ModelStatus(
            result_id=result_id
        )
    except Exception as e:
        raise HTTPException(status_code=0, detail=f"Error in predict model: {str(e)}")

@app.get('/status', response_model=SentimentResponse, summary='Резлуьтат модели')
async def model_status(result_id: str):
    '''
    Вывод результат модели
    
    **result_id**: ид запроса
    '''
    if result_id not in response_storage:
        raise HTTPException(
            status_code=404,
            detail="Result not found"
        )
    return response_storage[result_id]
