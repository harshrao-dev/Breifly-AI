from huggingface_hub import login
from transformers import T5ForConditionalGeneration, T5Tokenizer

login()

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

model.push_to_hub("text-summarizer-t5")
tokenizer.push_to_hub("text-summarizer-t5")