import ollama
import logging
import chromadb


dataset = []
client = chromadb.Client()

collection = client.create_collection(name="us-presidents-facts")

#read data.txt and add it to dataset array 
with open("data.txt", "r") as file:
    dataset = file.readlines()
    for i,line in enumerate(dataset):
        clean_text = line.replace("\n", "")
        dataset[i] = clean_text    

##create an embedding for each one of the facts
for i,line in enumerate(dataset):
    response = ollama.embed(
        model='mxbai-embed-large',
        input=line
    )
    embeddings = response["embeddings"]
    collection.add(
        documents=[line],
        ids=[str(i)],
        embeddings=embeddings
    )

#generate response

def generate_model_response(prompt,data):
    instruction_prompt = f"Use only the following pieces of context to answer the question: {data}"

    output = ollama.chat(
            model= 'llama3.2:1b',
           messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': prompt},
    ],
    stream=True
    )
    return output


while(True):

    userInput = input("Ask a question \n")

    ##create an embedding for the users prompt
    inputEmbedding = ollama.embed(
        model='mxbai-embed-large',
        input = userInput
    )

    results = collection.query(
        query_embeddings=inputEmbedding["embeddings"],
        n_results=5
    )
    data = results['documents'][0][0]
    output = generate_model_response(userInput,data)
    
    for r in output:
        print(r.message.content + "\n")
