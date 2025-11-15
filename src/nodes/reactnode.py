from typing import List, Optional
from src.state.state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains the node functions for the RAG workflow."""

    def __init__(self, llm, retriever):
        """Initializes RAG nodes.

        Args:
            llm: Language model instance (e.g. ChatOpenAI or similar).
            retriever: Document retriever instance.
        """
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy initialization of agent

    # ---- LangGraph node: retrieval ----
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve documents and update state."""
        docs: List[Document] = self.retriever.invoke(state.query)

        return RAGState(
            query=state.query,
            retrieved_docs=docs,
            response=state.response,
        )

    # ---- Tools for the agent ----
    def _build_tools(self) -> List[Tool]:
        """Build retriever + Wikipedia tools."""

        def retriever_tool_fn(query: str) -> str:
            """Fetch relevant passages from the indexed vectorstore."""
            docs: List[Document] = self.retriever.invoke(query)

            if not docs:
                return "No documents found."

            merged_chunks = []
            for i, d in enumerate(docs[:8], start=1):
                meta = getattr(d, "metadata", {}) or {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged_chunks.append(f"[{i}] {title}\n{d.page_content}")

            return "\n\n".join(merged_chunks)

        retriever_tool = Tool(
            name="retriever",
            func=retriever_tool_fn,
            description="Fetch passages from the indexed vector store.",
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wiki.run,
        )

        return [retriever_tool, wikipedia_tool]

    # ---- Build the ReAct agent using create_agent ----
    def _build_agent(self) -> None:
        """Create ReAct-style agent with tools using create_agent."""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer the 'retriever' tool for questions about the user's indexed documents. "
            "Use 'wikipedia' only for general world knowledge. "
            "Think step by step with tools, but return only the final helpful answer to the user."
        )

        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    # ---- LangGraph node: answer generation via agent ----
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate response using the create_agent ReAct agent."""
        if self._agent is None:
            self._build_agent()  # <- actually call it

        # create_agent expects state with a "messages" key
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=state.query)]}
        )

        # create_agent returns a dict with "messages" (AgentState)
        messages = result.get("messages", []) if isinstance(result, dict) else [result]

        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            query=state.query,
            retrieved_docs=state.retrieved_docs,
            response=answer or "Could not generate response.",
        )
