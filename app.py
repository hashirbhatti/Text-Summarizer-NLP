from pathlib import Path
import sys
from flask import Flask, render_template, redirect, Response, request

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from textSummarizer.pipeline.prediction import PredictionPipeline
import os

# Sample text for initialization
text: str = "What is Text Summarization?"

# Create Flask app instance
app = Flask(__name__)

prediction_pipeline = None

def get_prediction_pipeline():
    global prediction_pipeline
    if prediction_pipeline is None:
        prediction_pipeline = PredictionPipeline()
    return prediction_pipeline

@app.route("/", methods=["GET"])
def index():
    """
    Renders the index page.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Endpoint for making predictions using the text summarization model.
    """
    try:
        # Get the text from the POST request
        text = request.form.get("text")
        if not text:
            return Response("Please enter text to summarize.", status=400)

        # Use the singleton instance of the PredictionPipeline
        pipeline_obj = get_prediction_pipeline()

        # Make a prediction using the provided text
        summarized_text = pipeline_obj.predict(text)

        return summarized_text
    
    except Exception as e:
        return Response(f"Error Occurred: {e}")

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=8080)