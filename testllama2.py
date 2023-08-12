#Llama2 test
#test CUDA is installed
#LLama2 Inference
#https://colab.research.google.com/github/camenduru/text-generation-webui-colab/blob/main/llama-2-7b-chat.ipynb#scrollTo=gseGBLOtfNpL
#Cool ass anime chatbot

from transformers import AutoTokenizer
import transformers
import torch

#model = "meta-llama/Llama-2-7b-chat-hf"
model = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    #torch_dtype=torch.float16,
    device_map="auto",
    model_kwargs={"load_in_8bit": True}
    #model_kwargs={"load_in_4bit": True}
    
)

#prompt="How I mine for fish?"
prompt="You are an elf boy of the nighsilver woods, you are alone at home and awake hungry. Please list 3 steps you take when getting up"

#Read line form console
prompt = input("Enter a prompt: ")
while (len(prompt) > 0):

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        temperature=0.1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2000,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

    prompt = input("Enter a prompt: ")


