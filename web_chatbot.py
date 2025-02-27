import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model
model = GPT2LMHeadModel.from_pretrained("./chatbot_model")
tokenizer = GPT2Tokenizer.from_pretrained("./chatbot_model")

# Define chatbot function
def chatbot_response(message):
    input_ids = tokenizer.encode(message, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

# Gradio UI
iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Custom Chatbot")
iface.launch()