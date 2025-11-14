"""LangGraph Nodes for the Rag Workflow  """

from src.state.state import RAGState

class RAGNodes: 
    def __init__(self, llm , retriever):
        """ Initailizes RAG nodes 

        Args:
            llm : Language model instance
            retriever : Document retriever Instance
        """

        self.retriever= retriever
        self.llm = llm
        
    
    def retrieve_docs(self, state:RAGState) -> RAGState:
        """ Retrive documents from the vector store

        Args:
            state (RAGState): Current Rag State
            
        """
        docs = self.retriever.invoke(state.query)
        return RAGState(
            query = state.query, 
            retrieved_docs= docs            
        )
    
    def generate_answer(self, state: RAGState) -> RAGState: 
        """ Genrate response form retrieved 

        Args:
            state (RAGState): current RAG state of the workflow graph

        Returns:
            RAGState: Updated state of the RAG workflow graph
        """
        
        context = "/n/n".join([doc.page_content for doc in state.retrieved_docs])
        prompt = f"""
        Answer the query based on the context provided
        
        Context: 
            {context}
            
        query: {state.query}
        
        """
        response= self.llm.invoke(prompt)
        return RAGState(
            query= state.query,
            retrieved_docs= state.documents, 
            response= response.content
        )
        
        