import ollama
import logging
import chromadb
import input_data
import os

q = input_data.x

strings = []
embedding_questions = []
dataset = []
client = chromadb.Client() 
collection = client.create_collection(name="us-presidents-facts")

def parse_input():
    for i in q["questions"]:
        form_string = ""
        embedding_string = ""
        for j in i :
            
            if i[j] == "":
                form_string += "  ___ "
            else:
                form_string += " "
                form_string += i[j]
                embedding_string += " "
                embedding_string += i[j]
        strings.append(form_string)
        embedding_questions.append(embedding_string)

def embed_data():
    #read data.txt and add it to dataset array 
    with open("data.txt", "r") as file:
        dataset = file.readlines()
        for i,line in enumerate(dataset):
            clean_text = line.replace("\n", "")
            dataset[i] = clean_text    

    # ##create an embedding for each one of the facts
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
def generate_model_response_chat(prompt,data):
    instruction_prompt = f"""
    You are given context: {data}

    Task: Fill in the blank in the user's sentence using only the context provided. 
    Return **only** the word or phrase that completes the blank — no explanations.

    User Prompt: "{prompt}"
    Answer:"""

    output = ollama.chat(
            model= 'llama3.2:1b',
           messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': prompt},
    ],
    )
    return output


def RAG_SYSTEM_FLOW():

    embed_data()
    parse_input()
    outputList = {}

    i =0
    while( i < len(embedding_questions)):
        userInput = embedding_questions[i]
        inputEmbedding = ollama.embed(
                model='mxbai-embed-large',
                input = userInput
        )

        results = collection.query(
                query_embeddings=inputEmbedding["embeddings"],
                n_results=5
        )

        data = results['documents'][0][0]
        output = generate_model_response_chat(strings[i],data)
        outputList[i] = {"id":i,"input": strings[i], "output": output["message"]["content"]}
        i +=1
    
    return outputList

def log_output():

    outputList = RAG_SYSTEM_FLOW()

    log_dir = os.path.join(os.getcwd(), "logs")
    log_fname = os.path.join(log_dir, 'RAG_output.log')

    logging.basicConfig(level=logging.INFO, filename=log_fname,filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
        

    for i in outputList.values():
            logging.info(i)

if __name__ == "__main__":
    log_output()
    print("Script executed directly")