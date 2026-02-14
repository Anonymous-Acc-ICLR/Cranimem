# File: cranimem/agent/workflow.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    gating_node, retrieval_node, reasoning_node, 
    episodic_memory_node, consolidation_node
)

def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("gating", gating_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("episodic_store", episodic_memory_node)
    workflow.add_node("consolidation", consolidation_node)
    
    workflow.set_entry_point("gating")
    workflow.add_edge("gating", "retrieval")
    workflow.add_edge("retrieval", "reasoning")
    workflow.add_edge("reasoning", "episodic_store")
    workflow.add_edge("episodic_store", "consolidation")
    workflow.add_edge("consolidation", END)
    
    return workflow.compile()