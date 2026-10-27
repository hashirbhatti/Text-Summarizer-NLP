# Text-Summarizer-NLP

Text-Summarizer-NLP is an end-to-end Machine Learning Operations (MLOps) project that builds a production-ready NLP text summarization system. The project implements a complete ML lifecycle — from data ingestion and validation, through model training and evaluation — packaged as both an automated training pipeline and a user-facing web application.

## What Does This Project Do?

The system automatically summarizes long textual content (such as news articles or meeting transcripts) into concise, coherent summaries using a fine-tuned Pegasus sequence-to-sequence model. It follows a multi-stage ML pipeline:

1. **Data Ingestion** — Downloads and extracts a summary dataset from a remote source.  
2. **Data Validation** — Verifies that all required data splits (train, test, validation) are present.  
3. **Data Transformation** — Tokenizes the dataset using the Pegasus tokenizer and prepares it for training.  
4. **Model Training** — Fine-tunes the `google/pegasus-cnn_dailymail` model on the samsum dataset.  
5. **Model Evaluation** — Evaluates the trained model using ROUGE metrics and saves the results.  

A Flask-based web app provides a simple frontend where users can paste text and receive an instant summary powered by the trained model.

## Features

- **Full ML Pipeline** — Automates the entire training workflow with 5 configurable stages
- **ROUGE Evaluation** — Measures summary quality using ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum metrics
- **Web Interface** — Simple, responsive UI for text summarization via a browser
- **MLOps Best Practices** — Configuration-driven with YAML files, structured logging, and artifact management
- **GPU Support** — Built to leverage NVIDIA GPU acceleration with PyTorch CUDA support

## Project Structure

```
Text-Summarizer-NLP/
├── app.py                      # Flask web application
├── main.py                     # ML training pipeline (runs all 5 stages)
├── template.py                 # Code generation template
├── setup.py                    # Package setup configuration
├── requirements.txt            # Python dependencies
├── params.yaml                 # Training hyperparameters
├── config/
│   └── config.yaml             # Project configuration (paths, URLs, model names)
├── src/
│   └── textSummarizer/
│       ├── __init__.py
│       ├── config/
│       │   └── configuration.py    # Config reader and manager
│       ├── entity/
│       │   └── ...                # Data classes for configs
│       ├── utils/
│       │   └── common.py          # Shared utilities (YAML reader, directory creator)
│       ├── logging/
│       │   └── ...                # Custom logger setup
│       └── pipeline/
│           ├── stage_01_data_ingestion.py
│           ├── stage_02_data_validation.py
│           ├── stage_03_data_transformation.py
│           ├── stage_04_model_trainer.py
│           ├── stage_05_model_evaluation.py
│           └── prediction.py      # Web app prediction logic
├── templates/
│   └── index.html               # Flask HTML template (web UI)
├── artifacts/                   # Generated data, models, and metrics (gitignored)
├── logs/                        # Application logs (gitignored)
└── .venv/                       # Conda virtual environment (gitignored)
```

## Prerequisites

Before setting up the project locally, ensure you have the following installed:

- **Python 3.11** or higher
- **Git**
- **Conda** (Miniconda or Anaconda) — [Download conda](https://docs.anaconda.com/miniconda/)
- **NVIDIA GPU** (optional, recommended) with CUDA drivers for GPU-accelerated training

## Installation & Setup (2 Steps)

### Step 1: Clone the Repository and Set Up the Conda Environment

```bash
# Clone the repository
git clone https://github.com/hashirbhatti/Text-Summarizer-NLP.git
cd Text-Summarizer-NLP

# Create and activate the conda environment
conda create -p .venv python=3.11 -y
conda activate .venv
```

### Step 2: Install Dependencies and Run

```bash
# Install all Python dependencies
pip install -r requirements.txt
```

That's it. The environment is ready.

**Note:** If you encounter an SSL certificate error on Windows when installing packages, ensure `conda activate .venv` properly switches your Python to the `.venv` environment by checking `python --version` — it should show `Python 3.11.x`, not `3.14.x` or `3.8.x`. If not, restart your terminal and try activating the environment again.

## How to Run the ML Training Pipeline

The training pipeline runs all 5 stages sequentially (ingestion → validation → transformation → training → evaluation):

```bash
python main.py
```

What happens:
- **Stage 1 — Data Ingestion**: Downloads the samsum dataset and saves it to `artifacts/data_ingestion/`
- **Stage 2 — Data Validation**: Checks that train/test/validation splits exist
- **Stage 3 — Data Transformation**: Tokenizes the data using the Pegasus tokenizer
- **Stage 4 — Model Trainer**: Fine-tunes Pegasus on the dataset (this takes the longest)
- **Stage 5 — Model Evaluation**: Computes ROUGE scores and saves metrics to `artifacts/model_evaluation/metrics.csv`

All model artifacts (checkpoints, tokenizer, metrics) are saved under the `artifacts/` directory.

## Model Evaluation Results

The model was trained on **14,732** records from the samsum dataset and evaluated using ROUGE metrics:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.018737 |
| ROUGE-2 | 0.000341 |
| ROUGE-L | 0.018644 |
| ROUGE-Lsum | 0.018675 |

Full metrics are saved in `artifacts/model_evaluation/metrics.csv`.

## How to Start the Web Application

First, make sure the ML pipeline has been run at least once (so a trained model exists), then start the Flask web app:

```bash
python app.py
```

The server will start on **http://localhost:8080**. Open this URL in your browser to access the summarization web app.

The web app lets you:
- Paste any text into the input field
- Click **Summarize** to generate a summary
- View the generated summary below the input

The app uses the trained model from `artifacts/model_trainer/pegasus-samsum-model` for inference.

## Configuration

All key settings are managed through configuration files:

| File | Purpose |
|------|---------|
| `config/config.yaml` | Data paths, model checkpoint name, tokenizer name |
| `params.yaml` | Training hyperparameters (batch size, learning rate, epochs, etc.) |
| `requirements.txt` | Python package dependencies |

To customize the model, change `model_ckpt` and `tokenizer_name` in `config/config.yaml` (e.g., `google/pegasus-xsum` for XSum summarization).

## Troubleshooting

**`ModuleNotFoundError: No module named 'box'`**  
Make sure you've activated the `.venv` environment (`conda activate .venv`) and installed requirements (`pip install -r requirements.txt`).

**`ImportError: cannot import name 'DTensor'`**  
Your installed `transformers` package is too new for your PyTorch version. Downgrade with: `pip install transformers==4.44.0`

**SSL errors on Windows**  
Your Python environment may have a known SSL bug. Recreate the conda environment with Python 3.11+:  
```bash
conda env remove -p .venv -y
conda create -p .venv python=3.11 -y
conda activate .venv
pip install -r requirements.txt
```

**Flask not starting**  
Ensure `markupsafe`, `Jinja2`, and `werkzeug` are installed. Force reinstall if needed:  
```bash
pip install --force-reinstall markupsafe Jinja2 werkzeug flask
```

## Demo

https://github.com/user-attachments/assets/4d93135f-895d-4699-b532-9a6f2439bd41

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

## Author

**Hashir Bhatti**  
**Email:** bhattihashir26@gmail.com
