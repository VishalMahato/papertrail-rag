"""Vector store module for document embedding and retrieval."""

from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    """Manages a FAISS-based vector store for document retrieval."""

    def __init__(self, embedding_model: Optional[OpenAIEmbeddings] = None) -> None:
        """
        Initialize the VectorStore.

        Args:
            embedding_model: Optional custom embeddings instance.
                             Defaults to `OpenAIEmbeddings()`.
        """
        self.embedding = embedding_model or OpenAIEmbeddings()
        self.vectorstore: Optional[FAISS] = None
        self.retriever: Optional[BaseRetriever] = None

    def create_retriever(self, documents: List[Document], k: int = 4) -> None:
        """
        Create a vector store and retriever from documents.

        Args:
            documents: List of documents to embed and index.
            k: Default number of documents to retrieve per query.
        """
        if not documents:
            raise ValueError("Cannot create vector store with an empty document list.")

        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_retriever(self) -> BaseRetriever:
        """
        Get the retriever instance.

        Returns:
            The retriever instance.

        Raises:
            ValueError: If the vector store / retriever has not been initialized.
        """
        if self.retriever is None:
            raise ValueError(
                "Vector store is not initialized. Call 'create_retriever' first."
            )
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query string.
            k: Number of documents to retrieve.

        Returns:
            List of retrieved documents.

        Raises:
            ValueError: If the vector store / retriever has not been initialized.
        """
        if self.retriever is None:
            raise ValueError(
                "Vector store is not initialized. Call 'create_retriever' first."
            )


        self.retriever.search_kwargs["k"] = k
        return self.retriever.get_relevant_documents(query)
