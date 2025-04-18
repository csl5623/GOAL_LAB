# Ollama Chatbot with History Logging

## Overview

This is a Python-based chatbot project utilizing **Ollama's LLaMA3.2:1B** language model. The chatbot maintains a running conversation history and logs user inputs, model responses, and response times. It demonstrates how to interact with a language model while capturing timing information and storing the dialogue in a log file.

---

## Features

- Interactive chatbot using Ollama
- Maintains chat history for context-aware responses
- Logs all user inputs, responses, and duration metadata
- Tracks total response generation time in nanoseconds, seconds, and minutes
- Graceful exit using `exit` command

---

## Technologies Used

- **Python**
- **Ollama** (Language model backend)
- **Logging** module (to track interaction history and performance)


## Requirements
Python 3.7+

Ollama Python SDK (pip install ollama)

Access to the llama3.2:1b model in Ollama

## How to Run
Make sure you have the Ollama server running locally.

1. Install the Ollama Python package:
    pip install ollama
2. Run the script:
    python llama_chatbot.py

## Log Output
All logs are saved to chatbot.log, including:

- User prompts

- Assistant responses

- Total duration in nanoseconds, seconds, and minute