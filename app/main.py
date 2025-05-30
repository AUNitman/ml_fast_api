from fastapi import FastAPI

from ml.model import load_model

model = None

app = FastAPI()

@app.get('/')
def index():
    return {'text': 'somthing'}

@app.lifespan('startup')
def startup_event():
    global model
    model = load_model()
    
@app.get('/predict')
def predict_model(text: str):
    predict = model(text)
    
