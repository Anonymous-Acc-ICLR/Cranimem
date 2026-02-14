# File: cranimem/agent/state.py
from typing import TypedDict, Annotated, List, Any, Union, Optional
import operator

def buffer_reducer(current: Optional[List[Any]], update: Union[List[Any], str, None]) -> List[Any]:
    """
    Robust custom reducer for the Episodic Buffer.
    
    1. Ensures 'current' is always a list to prevent NoneType addition errors.
    2. If update is "CLEAR", it resets the short-term episodic buffer.
    3. Handles list-based updates by appending them to the current state.
    """

    if current is None:
        current = []
        
    if isinstance(update, str) and update == "CLEAR":
        return []
        
    if isinstance(update, list):
        return current + update
        
    return current

class AgentState(TypedDict):
    """
    Represents the state of the CraniMEM agent.
    Inherits the 'Internal State' (IS) concept from MEM1.
    """

    messages: Annotated[List[Any], operator.add]
    
    episodic_buffer: Annotated[List[Any], buffer_reducer] 
    
    current_goal: str       
    goal_text: str
    goal_vector: Any
    turn_count: int         
    current_input: str
    
    retrieved_context: str
    gating_result: dict
    agent_response: str
