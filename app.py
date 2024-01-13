from flask import Flask, render_template, redirect, Response, request
from textSummarizer.pipeline.prediction import PredictionPipeline
import os

# Sample text for initialization
text: str = "What is Text Summarization?"

# Create Flask app instance
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """
    Renders the index page.
    """
    return render_template("index.html")

@app.route("/train", methods=["GET"])
def training():
    """
    Endpoint for triggering training process.
    """
    try:
        # Execute the training script
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Endpoint for making predictions using the text summarization model.
    """
    try:
        # Get the text from the POST request
        text = request.form.get("text")

        # Create an instance of the PredictionPipeline
        obj = PredictionPipeline()

        # Make a prediction using the provided text
        summarized_text = obj.predict(text)

        return summarized_text
    
    except Exception as e:
        return Response(f"Error Occurred: {e}")

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=8080)
