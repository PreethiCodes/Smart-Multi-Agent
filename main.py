# -*- coding: utf-8 -*-



import os
from datetime import datetime
from typing import TypedDict

# optional dotenv support for local development
from dotenv import load_dotenv

# load environment variables from a .env file if present
load_dotenv()

import wikipedia
import qrcode
from gtts import gTTS
from langdetect import detect

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ------------------------------------------------------------
# 1. SETUP LLM (GROQ)
# ------------------------------------------------------------

# fetch the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set in environment")
os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(
    model="mixtral-8x7b-32768"
)

# ------------------------------------------------------------
# 2. TOOL DEFINITIONS
# ------------------------------------------------------------

def wikipedia_tool(query: str):
    """Fetch summary from Wikipedia"""
    try:
        return wikipedia.summary(query, sentences=5)
    except Exception as e:
        return f"Error: {str(e)}"


def sentiment_tool(text: str):
    """LLM-based sentiment analysis"""
    prompt = f"Classify the sentiment (Positive, Negative, Neutral): {text}"
    return llm.invoke(prompt).content


def language_tool(text: str):
    """Detect language"""
    try:
        return detect(text)
    except:
        return "unknown"


def qr_tool(data: str):
    """Generate QR code"""
    os.makedirs("generated_qr", exist_ok=True)
    filename = f"qr_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    filepath = os.path.join("generated_qr", filename)

    img = qrcode.make(data)
    img.save(filepath)

    return f"QR Code saved at {filepath}"


def email_tool(message: str):
    """Format text into professional email"""
    prompt = f"""
    Convert the following into a professional email with subject, greeting, and closing:

    {message}
    """
    return llm.invoke(prompt).content


def tts_tool(text: str):
    """Convert text to speech"""
    os.makedirs("generated_audio", exist_ok=True)
    filename = f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
    filepath = os.path.join("generated_audio", filename)

    tts = gTTS(text=text)
    tts.save(filepath)

    return f"Audio saved at {filepath}"


# ------------------------------------------------------------
# 3. STATE STRUCTURE
# ------------------------------------------------------------

class AgentState(TypedDict):
    input: str
    output: str
    next: str


# ------------------------------------------------------------
# 4. ROUTER NODE (IMPORTANT FIXED VERSION)
# ------------------------------------------------------------

def router_node(state: AgentState):
    query = state["input"]

    routing_prompt = f"""
    You are a routing controller.
    Choose exactly ONE action from the list.

    Valid actions (return ONLY one word):
    wikipedia
    sentiment
    language
    qr
    email
    tts

    Query: {query}

    Return only one valid action word.
    """

    decision = llm.invoke(routing_prompt).content.strip().lower()

    # ----------- Fuzzy matching safety layer -----------
    if "qr" in decision:
        decision = "qr"
    elif "email" in decision:
        decision = "email"
    elif "sentiment" in decision:
        decision = "sentiment"
    elif "language" in decision:
        decision = "language"
    elif "wiki" in decision:
        decision = "wikipedia"
    elif "tts" in decision or "speech" in decision:
        decision = "tts"

    # ----------- Final fallback to avoid LangGraph errors -----------
    valid = ["wikipedia", "sentiment", "language", "qr", "email", "tts"]
    if decision not in valid:
        decision = "wikipedia"

    return {"next": decision}


# ------------------------------------------------------------
# 5. TOOL NODES
# ------------------------------------------------------------

def wikipedia_node(state: AgentState):
    return {"output": wikipedia_tool(state["input"])}

def sentiment_node(state: AgentState):
    return {"output": sentiment_tool(state["input"])}

def language_node(state: AgentState):
    lang = language_tool(state["input"])
    return {"output": f"Detected Language: {lang}"}

def qr_node(state: AgentState):
    return {"output": qr_tool(state["input"])}

def email_node(state: AgentState):
    return {"output": email_tool(state["input"])}

def tts_node(state: AgentState):
    return {"output": tts_tool(state["input"])}


# ------------------------------------------------------------
# 6. BUILD LANGGRAPH
# ------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("wikipedia", wikipedia_node)
graph.add_node("sentiment", sentiment_node)
graph.add_node("language", language_node)
graph.add_node("qr", qr_node)
graph.add_node("email", email_node)
graph.add_node("tts", tts_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "wikipedia": "wikipedia",
        "sentiment": "sentiment",
        "language": "language",
        "qr": "qr",
        "email": "email",
        "tts": "tts",
    }
)

# End each tool node
graph.add_edge("wikipedia", END)
graph.add_edge("sentiment", END)
graph.add_edge("language", END)
graph.add_edge("qr", END)
graph.add_edge("email", END)
graph.add_edge("tts", END)

# Compile the graph
app = graph.compile()

# ------------------------------------------------------------
# 7. TEST RUN
# ------------------------------------------------------------

response = app.invoke({"input": "Generate QR code for https://google.com"})
print(response["output"])