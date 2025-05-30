import pytest

from ml.model import ClassifierText, load_model

@pytest.fixture(scope='function')
def model():
    return load_model()

@pytest.mark.parametrize(
    'text, expect_label',
    [
        ('bad', 'плохо'),
        ('good', 'хорошо'),
        ('netural', 'по-разному'),
    ],
)

def test_text(model, text:str, expected_label: str):
    model_pred = model(text)
    assert isinstance(model_pred, ClassifierText)
    assert model_pred.label == expected_label