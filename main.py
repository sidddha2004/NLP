# main.py - Vercel deployment version (No OpenAI required)

import os
from typing import List
import time

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Document processing
import PyPDF2
import docx

# AI and vector database
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Use sentence-transformers instead of OpenAI for embeddings (free alternative)
from sentence_transformers import SentenceTransformer

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG API with Gemini", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
gemini_model = None
embedding_model = None
pc = None
index = None

# Initialize AI models and services
def initialize_services():
    global gemini_model, embedding_model
    
    if gemini_model is None:
