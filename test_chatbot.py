from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./chatbot_model")
tokenizer = GPT2Tokenizer.from_pretrained("./chatbot_model")

# Ensure tokenizer has a pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Chat Function
def chat_with_bot():
    print("Chatbot is ready! Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        
        # Tokenize and move inputs to the correct device
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=100)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to GPU

        # Generate response
        output = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],  # Explicitly passing attention mask
            max_length=100, 
            pad_token_id=model.config.eos_token_id,  # Prevents errors
            temperature=0.7,  # Controls randomness
            top_p=0.9,  # Nucleus sampling for diverse responses
            do_sample=True  # Enables sampling
        )

        # Properly decode response
        bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"Chatbot: {bot_response}")

# Start chatbot
chat_with_bot()
