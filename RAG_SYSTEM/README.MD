# RAG System 

## Overview

This project demonstrates a basic **Retrieval-Augmented Generation (RAG)** system built using **Python**, **Ollama**, and **ChromaDB**. The system is designed to answer questions specifically about **U.S. presidents** using only the predefined knowledge provided to the Ollama model. There are two version of the exploration : Manual and Automated Mode

## Manual Mode

A terminal-based program that prompts the user to ask questions about U.S. Presidents. Ollama responds using only the predefined knowledge, but the responses are unstructured and conversational.



## Automated Mode

In this mode, input is provided via the input_data.py file as a JSON object. Each entry contains a question and a blank value for the answer. The system processes all questions automatically using the RAG pipeline—no user interaction is required.

---

## Installation Requirements

ollama pull mxbai-embed-large
pip install ollama
pip install ollama chromadb
ollama run llama3.2:1b

## Running Script

## Manual Mode

Run manual_program.py file 

## Automated MOde
Run automated_version.py

