from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    confidence_score: float
    requires_human: bool

def evaluate_confidence(state: AgentState):
    """
    Política de supervisión humana: Si el modelo no está seguro (score < 0.8),
    no responde al socio. Escala automáticamente a un operador humano.
    """
    if state['confidence_score'] < 0.8:
        return {"requires_human": True, "messages": ["Escalando a humano..."]}
    return {"requires_human": False}

# Nota: Este es el esqueleto del grafo. Los nodos de retriever y generator 
# se conectan aquí usando LangGraph para orquestar el flujo.
def build_agent_graph():
    workflow = StateGraph(AgentState)
    # workflow.add_node(...)
    # workflow.add_edge(...)
    return workflow.compile()
