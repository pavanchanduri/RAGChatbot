from langchain_groq import ChatGroq

MODEL_NAME = 'llama-3.3-70b-versatile'
model = ChatGroq(model_name=MODEL_NAME,
                temperature=0.4) # controls creativity

response = model.invoke("What is hugging face?")
print(response.content)