from ollama import chat
import logging

class ChatResponse():
    def __init__(self,message,total_duration,creation_time):
        self.creation_time = creation_time
        self.total_duration = total_duration
        self.message = message
    
    def __str__(self):
        return f"Chat Response: {self.message}\nCreation Time: {self.creation_time}\nDuration: {self.total_duration}\n"

  
m = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
  {
    'role': 'assistant',
    'content': "The sky is blue because of the way the Earth's atmosphere scatters sunlight.",
  },
  {
    'role': 'user',
    'content': 'What is the weather in Tokyo?',
  },
  {
    'role': 'assistant',
    'content': 'The weather in Tokyo is typically warm and humid during the summer months, with temperatures often exceeding 30°C (86°F). The city experiences a rainy season from June to September, with heavy rainfall and occasional typhoons. Winter is mild, with temperatures rarely dropping below freezing. The city is known for its high-tech and vibrant culture, with many popular tourist attractions such as the Tokyo Tower, Senso-ji Temple, and the bustling Shibuya district.',
  },
]

def chatbot(m):
  totalDuration = 0
  while True:
    user_input = input('Chat with history:')
    logging.info(f"User Prompt: {user_input}\n")

    if (user_input.lower() == "exit"):
       return totalDuration
    
    response = chat(
      'llama3.2:1b',
      messages=m
      + [
        {'role': 'user', 'content': user_input},
      ],
    )
    # Add the response to the messages to maintain the history
    m += [
      {'role': 'user', 'content': user_input},
      {'role': 'assistant', 'content': response.message.content},
    ]
    print(response.message.content + '\n')
    answer = ChatResponse(response.message.content,response["total_duration"],response["created_at"])
    totalDuration += response["total_duration"]
    logging.info(f"Answer value\n {answer}")

def main():
  logging.basicConfig(level=logging.INFO, filename="chatbot.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
  
  total_time = chatbot(m)
  logging.info(f"Total time spent generating the response: in nanoseconds {total_time}\n")
  total_seconds = total_time /1000000000
  logging.info(f"Total time spent generating the response: in seconds {total_seconds}\n")
  logging.info(f"Total time spent generating the response: in minutes {total_seconds/60}\n")

main()