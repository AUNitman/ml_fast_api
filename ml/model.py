from transformers import pipeline
from dataclasses import dataclass

import yaml
from pathlib import Path

config_path = Path(__file__).parent / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
@dataclass 
class ClassifierText:
    label: str
    score: float
    
def load_model():
    model_hf = pipeline(config['task'], model=config['model'], device=-1)
    
    def model(text: str) -> ClassifierText:
        pred = model_hf(text)
        pred_best_class = pred[0]
        return ClassifierText(
            label=pred_best_class['label'],
            score=pred_best_class['score']
        )
    
    return model
