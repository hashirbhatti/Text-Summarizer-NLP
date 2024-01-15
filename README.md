
# Text-Summarizer-NLP

This project presents a comprehensive solution for Natural Language Processing (NLP) text summarization, integrating Machine Learning Operations (MLOps) principles to streamline the development and deployment lifecycle. The system is designed to efficiently summarize textual content through a multi-step pipeline, encompassing key components such as data ingestion, data validation, data transformation, model training, and model evaluation. The project incorporates a robust Continuous Integration/Continuous Deployment (CI/CD) pipeline for automated testing, model deployment, and updates. Additionally, a user-friendly web application has been developed to provide a simple interface for users to interact with and utilize the summarization capabilities.




## Run Locally

Follow these simple steps to get the Text Summarizer up and running on your local machine:

**Step 1: Clone the Repository** 

```bash
  git clone https://github.com/hashirbhatti/Text-Summarizer-NLP.git
```

**Step 2: Go to the project directory**

```bash
  cd Text-Summarizer-NLP
```

**Step 3: Create & Activate a Conda Environment**

```bash
  conda create -p venv python=3.8 -y
```
```bash
  conda activate venv/
```

**Step 4: Install Requirements**

```bash
  pip install -r requirements.txt
```

**Step 5: Run the Application**

```bash
  python app.py
```

Once the application is successfully launched, open your web browser and navigate to http://localhost:8080 to access the user-friendly web app for text summarization.

**NOTE:** The default port for the application is set to 8080. If you wish to use a different port, you can modify the port configuration in the app.py file. Simply locate the app.run() statement and update the port parameter to your desired port number. Save the file, and the application will run on the newly specified port upon the next execution.

## Demo

Insert gif or link to demo


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

The MIT License is a permissive open-source license that allows you to use, modify, and distribute the code for both commercial and non-commercial purposes. For more details, refer to the MIT License document.
## Author

**Hashir Bhatti**\
**Email:** bhattihashir26@gmail.com
