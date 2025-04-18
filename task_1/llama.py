import ollama
import logging
from task_1.data import questions


class AnswerResponse():
    def __init__(self,id,creation_time,total_duration,response):
        self.id = id
        self.creation_time = creation_time
        self.total_duration = total_duration
        self.response = response
    
    def __str__(self):
        return f"Id: {self.id}\nResponse: {self.response}\nCreation Time: {self.creation_time}\nDuration: {self.total_duration}\n"


def generate_response(q,llama3_21):
    question = q["question"]
    response = ollama.generate(model=llama3_21, prompt=question, stream=False)
    creation_time = response["created_at"]
    total_duration = response["total_duration"]
    r = response["response"]
    answer = AnswerResponse(q["id"],creation_time,total_duration,r)
    logging.info(f"Answer value\n {answer}")
    return total_duration
    # logging.info(total_duration)


def run_llama3_21B():
    llama3_21 = "llama3.2:1b"
    logging.basicConfig(level=logging.INFO, filename=f"{llama3_21}.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
    total_time = 0
    
    for q in questions:
        total_time += generate_response(q,llama3_21)
    logging.info(f"Total time spent generating the response: in nanoseconds {total_time}\n")
    total_seconds = total_time /1000000000
    logging.info(f"Total time spent generating the response: in seconds {total_seconds}\n")
    logging.info(f"Total time spent generating the response: in minutes {total_seconds/60}\n")


def llama_323B():
    total_time = 0
    llama3_2 = "llama3.2:3b"
    logging.basicConfig(level=logging.INFO, filename=f"{llama3_2}.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
    for q in questions:
        total_time += generate_response(q,llama3_2)
    logging.info(f"Total time spent generating the response: in nanoseconds {total_time}\n")
    total_seconds = total_time /1000000000
    logging.info(f"Total time spent generating the response: in seconds {total_seconds}\n")
    logging.info(f"Total time spent generating the response: in minutes {total_seconds/60}\n")
    
llama_323B()
