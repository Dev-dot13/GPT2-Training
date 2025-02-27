# Custom Chatbot Using GPT-2

## Project Overview
This chatbot model is trained on a custom dataset consisting of **1,000 prompt-response pairs** related to Indian heritage. The model is trained, tested, and deployed on the web. However, due to the **limited capability of GPT-2** and the **small dataset size**, the chatbot does not function properly. For effective performance, at least **50,000+ pairs** are recommended. The biggest challenge in this project was manually creating a **custom dataset** from scratch with a large number of high-quality prompt-response pairs.

## Technologies Used
- **Hardware:**
  - NVIDIA RTX 4060 GPU (Used for training acceleration)
  - 16GB RAM (Handles model loading and inference)
- **Software:**
  - Windows 11 24H2
  - Python (latest stable version recommended)
  - CUDA (for GPU acceleration, compatible with PyTorch)
- **Libraries & Packages:**
  - `torch` (For deep learning computations using PyTorch)
  - `transformers` (From Hugging Face, used for GPT-2 model)
  - `gradio` (For deploying and testing chatbot via a web UI)
  - **A `requirements.txt` file is provided** for installing all recommended libraries and packages.

## Setup Instructions
1. **Install Required Packages:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Ensure CUDA is Installed** (for GPU acceleration):
   - Install the latest CUDA version compatible with PyTorch.
   - Verify installation with:
     ```powershell
     nvcc --version
     ```

3. **Prepare the Model:**
   - Place your trained GPT-2 model and tokenizer inside `./chatbot_model/`.
   - Ensure the folder contains:
     - `pytorch_model.bin`
     - `config.json`
     - `vocab.json`
     - `merges.txt`

4. **Run the Chatbot:**
   ```powershell
   python chatbot.py
   ```
   
## Dataset
Dataset for this model was custom made, ai generated specifically on the topic of Indian Heritage. It is present in json format and not uploaded on github due to it's large size as it contains upto 1000 prompt-response pairs. Sample of the file is given below:
```
[
    {
        "prompt": "Which dynasty was responsible for Brihadeeswarar Temple?",
        "response": "It was built during the Chola dynasty."
    },
    {
        "prompt": "Who built Ajanta Caves?",
        "response": "Ajanta Caves was built by Buddhist Monks."
    },
    {
        "prompt": "Tell me an interesting fact about Charminar.",
        "response": "It was built to commemorate the end of a deadly plague in Hyderabad."
    }
]
```

## Chatbot Response Issues
The chatbot produces incoherent responses due to **insufficient training data and the limited capacity of GPT-2**. Here are some example interactions:

**Example 1:**
```
You: Who built Sun Temple?
Chatbot: Nar by Dev by Nar by Nar.
```
Correct answer: King Narasingha Deva I.
Explanation: The chatbot locks onto **keywords** from the input question and tries to recall related words from its limited dataset. Since "Narasingha Deva I" was mentioned in training but not sufficiently, it outputs fragmented words like "Nar" (Narasingha) and "Dev" (Deva).

**Example 2:**
```
You: Where is Brihadeeswarar Temple located?
Chatbot: Tamil Nadu Nadu Nadu.. Tamil Nadu.
```
Correct answer: Tamil Nadu.
Explanation: The chatbot correctly identifies "Tamil Nadu" as relevant but repeats words due to its lack of proper sentence formation. This happens because it tries to generate text based on **pattern recognition** rather than complete sentence understanding.

While the responses seem **incoherent**, they are **highly relevant** to the correct answers. The primary issue is the **small dataset size (1,000 pairs instead of the recommended 50,000+ pairs)**, which leads to improper sentence formation and repetition of keywords.

## Challenges & Limitations
- **GPT-2 Model Limitations:**
  - GPT-2 is not as advanced as newer models like GPT-3 or GPT-4, leading to suboptimal performance.
- **Insufficient Dataset Size:**
  - The model was trained on **only 1,000 prompt-response pairs**, which is far below the recommended **50,000+ pairs** for meaningful performance.
  - Due to the small dataset, the chatbot struggles with generating diverse and contextually accurate responses.
- **Dataset Creation Challenge:**
  - Manually creating a **50,000+ pair dataset** on a specific topic is time-consuming and requires extensive research.
- **Hardware Requirements:**
  - Training and running deep learning models require increasing **hardware capabilities**, such as **more powerful GPUs and higher RAM** for **faster and smoother processing** of large amounts of data.

## Accuracy and Testing
- There is no built-in accuracy score for GPT-2 text generation.
- The chatbot was manually tested, and results indicate that **the responses are not satisfactory due to limited training data**.
- Fine-tuning with a **larger dataset and a more advanced model** would improve accuracy.

## Deactivating Everything After Use
- Stop the chatbot manually by closing the terminal.
- Clear GPU memory by restarting the system or running:
  ```powershell
  taskkill /IM python.exe /F
  ```
