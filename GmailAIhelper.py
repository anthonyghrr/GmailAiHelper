from gpt4all import GPT4All
model = GPT4All("Llama-3.2-3B-Instruct-Q4_0.gguf") # downloads / loads a 4.66GB LLM
with model.chat_session():
    print(model.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024))