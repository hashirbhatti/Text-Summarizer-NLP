import os
from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.tokenizer is None or self.model is None:
            tokenizer_path = str(self.config.tokenizer_path)
            model_path = str(self.config.model_path)

            # Load tokenizer from tokenizer_path or model_path or base checkpoint
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            elif os.path.exists(model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

            # Load model from model_path or base checkpoint
            if os.path.exists(model_path):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")

    def predict(self, text):
        self._load_model()

        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        print("Dialogue:")
        print(text)

        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **gen_kwargs
        )
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        print("\nModel Summary:")
        print(output)

        return output
