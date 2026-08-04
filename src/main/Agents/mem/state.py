from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    question_id: int
    question: str 
    input_route: str
    # Evaluation-only switch. Production requests omit this field and therefore
    # continue to use the default hybrid + reranking path.
    retrieval_config: str

    search_targets: List[Dict[str, Any]] 
    plan: Dict[str, Any]
    planner_think: str
    reasoning_think: str

    retrieved_documents: List[Dict[str, Any]]

    extracted_evidence: str 
    relevant_chunk_ids: List[str]
    evidence_id_map: Dict[str, str]
    doc_registry: Dict[str, Dict[str, Any]]
    answer_check: Dict[str, Any]

    search_retries: int
