
# Groq-Powered Multi-Tool AI Orchestrator using LangGraph

## Overview
This project is a Multi-Tool AI Agent built using LangGraph and Groq’s Llama 3 LLM.  
The system automatically routes user queries to the correct tool and executes the action.

Supported tools:
- Wikipedia Search
- Sentiment Analysis
- Language Detection
- QR Code Generation
- Email Formatting
- Text-to-Speech Conversion

The agent uses an intelligent router to determine which tool to use based on the input query.

---

## Architecture

### Router
The router uses the LLM to analyze the user query and chooses exactly one action:
- wikipedia  
- sentiment  
- language  
- qr  
- email  
- tts  

A fuzzy-matching safeguard prevents invalid outputs.

### Tools
Each tool is handled in a dedicated node:
- **Wikipedia Tool** – Fetches 5-sentence summaries
- **Sentiment Tool** – Classifies sentiment as Positive, Negative, or Neutral
- **Language Tool** – Detects language using `langdetect`
- **QR Tool** – Generates QR codes using the `qrcode` library
- **Email Tool** – Formats text into a professional email
- **TTS Tool** – Converts text into speech using gTTS

---

## Tech Stack

| Component | Technology |
|----------|------------|
| LLM | Groq Llama3-8B |
| Framework | LangGraph |
| Tools | wikipedia, langdetect, qrcode, gTTS |
| Platform | Google Colab / Python |
| Routing | LLM-based one-word decision |

---

## Project Structure
