import ollama
import logging


dataset = []

#read data set
with open("data.txt", "r") as file:
    dataset = file.readlines()
    for i,line in enumerate(dataset):
        clean_text = line.replace("\n", "")
        dataset[i] = clean_text    

##use ollama embedding model to create embeddings for data

for i,line in enumerate(dataset):
    embedding = ollama.embed(
        model='mxbai-embed-large',
        input=line
    )["embeddings"]
    ##store to pgvector


## get query of user, embed it and query it from pg vector