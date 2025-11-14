""" Graph builder for LangGraph Workflow"""

from langgraph.graph import START, END , StateGraph
from src.state.state import RAGState
from src.nodes.nodes import RAGNodes

class GraphBuilder:
    """ Builds and Orchestrates the langgraph workflow"""
    
    def __init__(self,retriver,llm):
        """Initializes graph builder

        Args:
            retriver: Document retriever instance to retrieve from vector store
            llm : language model instance
        """
        self.nodes= RAGNodes
        self.graph= None
    
    def build(self):
        """
        Builds the rag workflow graph
        
        Returns: 
            Compiled graph instance
            
        """
        builder = StateGraph(RAGState)
        
        # adding nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("retriever", self.nodes.generate_answer)
        
        builder.set_entry_point("retriever")
        
        builder.add_edge("retriever","responder")
        builder.add_edge("responder", END)
        
        # compile graph 
        self.graph = builder.compile()
        return self.graph
        
        
    
    def run(self, query:str) -> dict:
        """ Run the rag workflow

        Args:
            query (str): user query

        Returns:
            dict: Final state with answer
            
        """
        if self.graph is None: 
            self.build()
        
        intial_state= RAGState(query=query)
        return self.graph.invoke(intial_state)