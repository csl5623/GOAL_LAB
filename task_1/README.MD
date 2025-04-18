# Ollama Model Evaluation using LLaMA3

## Overview

This project is designed to evaluate the performance and output of different LLaMA3 models (via Ollama) on a list of questions. It tracks the generation time, response content, and saves logs for further analysis.

---

## Features

- Uses **Ollama** to call LLaMA3 models (1B and 3B)
- Automatically loads and processes a list of questions from `task_1.data`
- Logs each response with metadata: ID, timestamp, duration, and full answer
- Calculates total response time (in nanoseconds, seconds, and minutes)
- Saves logs to separate files based on model used (`llama3.2:1b.log` or `llama3.2:3b.log`)

## Technologies Used

- **Python 3.7+**
- **Ollama** (`pip install ollama`)
- **Logging** module


## How to Run

1. Setup Ollama
- Make sure your Ollama server is running and supports:

- lama3.2:1b

- llama3.2:3b

2. Install Dependencies

- pip install ollama

3. Run the Script

- python program.py