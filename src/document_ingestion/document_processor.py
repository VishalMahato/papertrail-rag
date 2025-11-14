"""Document processing module for loading and splitting documents."""

from pathlib import Path
from typing import List, Sequence, Union
from urllib.parse import urlparse

from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        """Initialize DocumentProcessor.

        Args:
            chunk_size: Size of text chunks. Defaults to 500.
            chunk_overlap: Overlap between chunks. Defaults to 100.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a given URL.

        Args:
            url: HTTP or HTTPS URL to load the document(s) from.

        Returns:
            A list of loaded `Document` instances.

        Raises:
            ValueError: If the URL is empty or uses an unsupported scheme.
            RuntimeError: If loading the document(s) fails.
        """
        if not url:
            raise ValueError("Parameter 'url' must be a non-empty string.")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Unsupported URL scheme {parsed.scheme!r}. Only 'http' and 'https' are allowed."
            )

        loader = WebBaseLoader(url)

        try:
            return loader.load()
        except Exception as exc:
            raise RuntimeError(f"Failed to load documents from URL: {url!r}") from exc

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load all PDFs within a directory.

        Args:
            directory: Path of the directory.

        Returns:
            A list of loaded `Document` instances.

        Raises:
            ValueError: If the directory path is invalid or does not exist.
            RuntimeError: If loading the document(s) fails.
        """
        if not directory:
            raise ValueError("Parameter 'directory' must be a non-empty string or Path.")

        directory = Path(directory)

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Provided path is not a directory: {directory}")

        loader = PyPDFDirectoryLoader(path=directory)

        try:
            return loader.load()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PDF documents from directory: {directory!r}"
            ) from exc

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a text file.

        Args:
            file_path: Path of the file.

        Returns:
            A list of loaded `Document` instances.

        Raises:
            ValueError: If the path is invalid or does not exist.
            RuntimeError: If loading the document(s) fails.
        """
        if not file_path:
            raise ValueError("Parameter 'file_path' must be a non-empty string or Path.")

        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Provided path is not a file: {file_path}")
        if file_path.suffix.lower() != ".txt":
            raise ValueError(
                f"Unsupported file type for load_from_txt: {file_path.suffix!r} "
                f"(expected '.txt')."
                )

      
        loader = TextLoader(str(file_path))

        try:
            return loader.load()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load text document(s) from file: {file_path!r}"
            ) from exc

    def load_documents(self, sources: Sequence[Union[str, Path]]) -> List[Document]:
        """Load all documents from the provided sources.

        Args:
            sources: Iterable of URLs, file paths, or directory paths.

        Returns:
            A list of loaded `Document` instances.

        Raises:
            ValueError: If `sources` is empty, or a path does not exist,
                or the source type is unsupported.
            RuntimeError: Propagated from the underlying loader methods if they fail.
        """
        
        if not sources:
            raise ValueError("Parameter 'sources' must be a non-empty sequence.")

        docs: List[Document] = []

        for source in sources:
            if not source:
                continue

            source_str = str(source)

            # URL case
            if source_str.startswith(("http://", "https://")):
                loaded_docs = self.load_from_url(source_str)
                if loaded_docs:
                    docs.extend(loaded_docs)
                continue  

        
            path = Path(source)

            if not path.exists():
                raise ValueError(f"Source path does not exist: {path}")

            if path.is_dir():
                loaded_docs = self.load_from_pdf_dir(path)
                if loaded_docs:
                    docs.extend(loaded_docs)
            else:
                suffix = path.suffix.lower()

                if suffix == ".txt":
                    loaded_docs = self.load_from_txt(path)
                    if loaded_docs:
                        docs.extend(loaded_docs)
                else:
                    raise ValueError(
                        f"Unsupported file type: {suffix} (source: {path})"
                    )

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to be chunked.

        Returns:
            A list of chunked `Document` instances.
        """
        if not documents:
            return []

        return self.splitter.split_documents(documents)

    def process_url(self, urls: List[str]) -> List[Document]:
        """Complete pipeline to load and split documents from URLs.

        Args:
            urls: List of URLs to process.

        Returns:
            List of processed document chunks.
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)
