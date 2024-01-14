from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()

class PredictionPipelineDependency:
    def __init__(self):
        self.obj = PredictionPipeline()

    def get_pipeline(self):
        return self.obj

@app.post("/predict")
async def predict_route(data: dict, pipeline: PredictionPipeline = Depends(PredictionPipelineDependency().get_pipeline)):
    try:
        text = data.get("text", "")
        summary = pipeline.predict(text)
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        return JSONResponse(content={"error": f"Error Occurred! {e}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_backend:app", host="0.0.0.0", port=8000, reload=True)
