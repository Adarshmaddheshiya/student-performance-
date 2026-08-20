from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

import pandas as pd
import joblib

from pathlib import Path


# =====================================
# FastAPI Application
# =====================================

app = FastAPI(
    title="Student Performance Prediction API"
)


# =====================================
# Project Paths
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "data_analysis" / "Model.pkl"

STATIC_PATH = BASE_DIR / "app" / "static"


# =====================================
# Load Model
# =====================================

model = joblib.load(MODEL_PATH)


# =====================================
# Static Files
# =====================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_PATH),
    name="static"
)


# =====================================
# Input Schema
# =====================================

class StudentInput(BaseModel):

    gender: str

    race_ethnicity: str

    parental_level_of_education: str

    lunch: str

    test_preparation_course: str

    reading_score: float

    writing_score: float


# =====================================
# Home Page
# =====================================

@app.get("/", response_class=HTMLResponse)
def home():

    html_file = STATIC_PATH / "index.html"

    return html_file.read_text(
        encoding="utf-8"
    )


# =====================================
# Prediction API
# =====================================

@app.post("/predict")
def predict(data: StudentInput):

    input_data = pd.DataFrame([
        {
            "gender": data.gender,

            "race_ethnicity":
                data.race_ethnicity,

            "parental_level_of_education":
                data.parental_level_of_education,

            "lunch":
                data.lunch,

            "test_preparation_course":
                data.test_preparation_course,

            "reading_score":
                data.reading_score,

            "writing_score":
                data.writing_score
        }
    ])


    prediction = model.predict(input_data)


    return {
        "predicted_math_score":
            round(float(prediction[0]), 2)
    }