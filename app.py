from pathlib import Path
import sys
from flask import Flask, render_template, redirect, Response, request

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from textSummarizer.pipeline.prediction import PredictionPipeline

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


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the server is running.
    """
    return Response("OK", status=200, mimetype="text/plain")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Endpoint for making predictions using the text summarization model.
    """
    try:
        text = request.form.get("text", "").strip()
        if not text:
            return Response(
                "Please enter valid text to summarize.",
                status=400,
                mimetype="text/plain",
            )

        if len(text) < 50:
            return Response(
                "Text is too short to summarize. Please enter at least 50 characters.",
                status=400,
                mimetype="text/plain",
            )

        pipeline_obj = get_prediction_pipeline()
        summarized_text = pipeline_obj.predict(text)

        if not summarized_text or len(summarized_text.strip()) < 10:
            return Response(
                "Unable to generate summary. Please try with different text.",
                status=422,
                mimetype="text/plain",
            )

        return Response(summarized_text, status=200, mimetype="text/plain")

    except ValueError as e:
        return Response(f"Invalid input: {str(e)}", status=400, mimetype="text/plain")
    except Exception as e:
        return Response(f"Error Occurred: {str(e)}", status=500, mimetype="text/plain")


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=8080)
