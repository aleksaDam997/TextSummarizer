from src.textsummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        output_ids = model.generate(
            inputs["input_ids"],
            length_penalty=0.8,
            num_beams=8,
            max_length=128
        )

        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("Dialogue:")
        print(text)
        print("\nModel Summary:")
        print(output)

        return output