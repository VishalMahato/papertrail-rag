""" state definition for LangGraph """

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    """State object for RAG workflow"""
    query: str
    retrieved_docs: List[Document] =[]
    response:str = ""