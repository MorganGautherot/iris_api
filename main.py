from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Define input payload using Pydantic
class InputPayload(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define output payload using Pydantic
class OutputPayload(BaseModel):
    species: str

# Define class labels
species_dict = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# API Endpoint to predict flower species
@app.post("/predict", response_model=OutputPayload)
def predict_species(data: InputPayload):
    # Convert input into array
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    species = species_dict[prediction[0]]

    return {"species": species}

@app.post("/echo")
def echo_input(payload: InputPayload):
    # Print the received payload to the console
    print("Received Input:", payload.dict())

    # Return the same payload to the user
    return {"received_data": payload}