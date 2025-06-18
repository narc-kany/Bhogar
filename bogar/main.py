from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Bhogar Drug Discovery API is running."}

@app.get("/discover")
def discover():
    result = subprocess.run(["python", "app.py"], capture_output=True, text=True)
    return {"output": result.stdout.strip()}
