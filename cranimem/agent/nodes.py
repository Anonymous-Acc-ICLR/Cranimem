# File: cranimem/agent/nodes.py
import json
import re
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from cranimem.config import settings
from cranimem.core.graph_store import Neo4jNeocortex
from cranimem.core.embedding import get_embeddings
from cranimem.utils.llm_factory import get_llm
from cranimem.utils.json_utils import (
    safe_json_load,
    parse_json_response,
    normalize_utility_tag,
    normalize_entity_relation,
)
from cranimem.cognitive.prompts import (
    REASONING_PROMPT,
    UTILITY_TAGGING_PROMPT,
    REFLEX_UTILITY_PROMPT,
    CORTEX_GATING_PROMPT,
)
from ..cognitive.prompts import ENTITY_RELATION_EXTRACTION_PROMPT, ENTITY_EXTRACTION_PROMPT 

embedding_model = None
neocortex = None

llm_structured = None
llm_creative = None
utility_parser = JsonOutputParser()
utility_chain = None
entity_relation_parser = JsonOutputParser()
entity_relation_chain = None

import numpy as np
from numpy.linalg import norm

def robust_parse_tagger(llm_output):
    """Clean markdown and parse JSON safely."""
    text = llm_output.content if hasattr(llm_output, "content") else str(llm_output)
    try:
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        parsed = safe_json_load(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"importance": 0.0, "surprise": 0.0, "emotion": 0.0, "entities": []}

def robust_parse_cortex(llm_output):
    """Parse low-similarity gating output with a fallback structure."""
    text = llm_output.content if hasattr(llm_output, "content") else str(llm_output)
    try:
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        parsed = safe_json_load(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {
        "is_noise": True,
        "category": "noise",
        "importance": 0.0,
        "surprise": 0.0,
        "emotion": 0.0,
        "entities": [],
        "reason": "Parse fallback"
    }

def _ensure_llms():
    global llm_structured, llm_creative, utility_chain, entity_relation_chain
    if llm_structured is None:
        llm_structured = get_llm(temperature=0.0)
    if llm_creative is None:
        llm_creative = get_llm(temperature=0.7)
    if utility_chain is None:
        utility_chain = UTILITY_TAGGING_PROMPT | llm_structured | StrOutputParser()
    if entity_relation_chain is None:
        entity_relation_chain = ENTITY_RELATION_EXTRACTION_PROMPT | llm_structured | entity_relation_parser

def _ensure_neocortex():
    """
    Lazily initialize embeddings + Neo4j to avoid import-time hangs,
    especially in restricted environments (e.g., Kaggle).
    """
    global embedding_model, neocortex
    if embedding_model is None:
        embedding_model = get_embeddings()
    if neocortex is None:
        neocortex = Neo4jNeocortex(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD.get_secret_value(),
            embedding_model=embedding_model
        )


def configure_llms(structured_llm=None, creative_llm=None):
    """
    Allows external callers (e.g., Kaggle eval) to override the LLMs used in nodes.
    Rebuilds dependent chains to ensure the override takes effect.
    """
    global llm_structured, llm_creative, utility_chain, entity_relation_chain
    if structured_llm is not None:
        llm_structured = structured_llm
    if creative_llm is not None:
        llm_creative = creative_llm
    _ensure_llms()
    utility_chain = UTILITY_TAGGING_PROMPT | llm_structured | StrOutputParser()
    entity_relation_chain = ENTITY_RELATION_EXTRACTION_PROMPT | llm_structured | entity_relation_parser

def _safe_json_load(text: str):
    return safe_json_load(text)

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    if norm(v1) == 0 or norm(v2) == 0: 
        return 0.0
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def gating_node(state: Dict[str, Any]):
    """
    Streamlined Gating: Focuses ONLY on filtering.
    Delays entity extraction to the Consolidation phase.
    """
    _ensure_llms()
    _ensure_neocortex()
    user_input = state.get("current_input", "")
    gating_mode = state.get("gating_mode", settings.GATING_MODE)
    current_goal = state.get("current_goal", "")
    goal_vector = state.get("goal_vector")
    if goal_vector is None and isinstance(current_goal, (list, tuple)):
        goal_vector = current_goal
    
    if goal_vector is not None:
        goal_vec = np.array(goal_vector, dtype=float)
    else:
        projected_goal = f"Information related to {current_goal}"
        goal_vec = neocortex.embedder.embed_query(projected_goal)
    input_vec = neocortex.embedder.embed_query(user_input)
    sim_score = cosine_similarity(goal_vec, input_vec)
    
    print(f"VECTOR SCORE: {sim_score:.4f}")


    if sim_score >= settings.REFLEX_PASS_THRESHOLD:
        print("REFLEX: High similarity, utility-only tagging...")
        try:
            tag_raw = (REFLEX_UTILITY_PROMPT | llm_structured | StrOutputParser()).invoke({"input": user_input})
            tag = normalize_utility_tag(robust_parse_tagger(tag_raw))
        except Exception as e:
            print(f"Reflex tagging failed: {e}")
            tag = {"importance": 0.0, "surprise": 0.0, "emotion": 0.0, "entities": []}

        importance = tag["importance"]
        surprise = tag["surprise"]
        emotion = tag["emotion"]
        entities = tag["entities"] or []
        base_utility = max(0.0, min(1.0, (importance + surprise + emotion) / 3.0))

        return {
            "gating_result": {
                "is_noise": False,
                "priority_score": base_utility,
                "relevance_score": sim_score,
                "importance": importance,
                "surprise": surprise,
                "emotion": emotion,
                "base_utility": base_utility,
                "entities": entities,
                "reason": "Reflex: High similarity"
            }
        }

    print(" CORTEX: Low similarity, deciding relevance + utility...")
    try:
        cortex_raw = (CORTEX_GATING_PROMPT | llm_structured | StrOutputParser()).invoke({"input": user_input})
        cortex = robust_parse_cortex(cortex_raw)
        importance = float(cortex.get("importance", 0.0) or 0.0)
        surprise = float(cortex.get("surprise", 0.0) or 0.0)
        emotion = float(cortex.get("emotion", 0.0) or 0.0)
        entities = cortex.get("entities", []) or []
        is_noise = bool(cortex.get("is_noise", True))
        base_utility = max(0.0, min(1.0, (importance + surprise + emotion) / 3.0))
        reason = cortex.get("reason", "Cortex gating")
    except Exception as e:
        print(f"Cortex gating failed: {e}")
        importance = surprise = emotion = 0.0
        entities = []
        is_noise = True
        base_utility = 0.0
        reason = "Cortex fallback"

    return {
        "gating_result": {
            "is_noise": is_noise,
            "priority_score": base_utility,
            "relevance_score": sim_score,
            "importance": importance,
            "surprise": surprise,
            "emotion": emotion,
            "base_utility": base_utility,
            "entities": entities,
            "reason": reason
        }
    }
        


def retrieval_node(state: Dict[str, Any]):
    """
    Hybrid Retrieval: Combines Neo4j (Long-Term) and Episodic Buffer (Short-Term).
    This provides the LLM with full context of both 'facts' and 'chitchat'.
    """
    _ensure_neocortex()
    long_term = neocortex.retrieve_relevant(state["current_input"])
    
    short_term = [m["content"] for m in state.get("episodic_buffer", [])]
    
    combined_context = "--- LONG-TERM MEMORIES ---\n" + "\n".join(long_term) + \
                       "\n\n--- RECENT WORKING MEMORY ---\n" + "\n".join(short_term)
    
    return {"retrieved_context": combined_context}

def reasoning_node(state: Dict[str, Any]):
    """MEM1 reasoning using the hybrid context."""
    _ensure_llms()
    chain = REASONING_PROMPT | llm_creative | StrOutputParser()
    goal_text = state.get("goal_text", state.get("current_goal", ""))
    if not isinstance(goal_text, str):
        goal_text = str(goal_text)
    context_text = state.get("retrieved_context", "")
    max_chars = getattr(settings, "REASONING_MAX_CONTEXT_CHARS", None)
    if max_chars is not None:
        max_chars = int(max_chars)
        if max_chars > 0 and len(context_text) > max_chars:
            print(f"GPU SAFETY: Truncating context from {len(context_text)} to {max_chars} characters.")
            context_text = context_text[:max_chars] + "\n... [Context Truncated for Memory Safety]"
    raw_output = chain.invoke({
        "current_goal": goal_text,
        "context": context_text,
        "current_input": state.get("current_input", "")
    })
    
    match = re.search(r"<RESPONSE>(.*?)</RESPONSE>", raw_output, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
    else:
        final_answer = raw_output
        for marker in ("System:", "Context:", "User Input:"):
            idx = final_answer.rfind(marker)
            if idx != -1:
                final_answer = final_answer[idx + len(marker):]
        if "Assistant:" in final_answer:
            final_answer = final_answer.rsplit("Assistant:", 1)[-1]
        final_answer = final_answer.strip()
    
    return {"agent_response": final_answer}

def episodic_memory_node(state: Dict[str, Any]):
    """
    The 'Working Memory' Bucket: Saves EVERY turn to the buffer.
    
    Crucially, it increments 'turn_count' every time to ensure the 
    consolidation_node triggers periodically as planned.
    """
    gating = state.get("gating_result", {})
    
    memory_item = {
        "content": f"User: {state.get('current_input')}\nAgent: {state.get('agent_response')}",
        "utility_score": gating.get("priority_score", 0.0),
        "is_noise": gating.get("is_noise", True),
        "entities": gating.get("entities", []),
        "metadata": {
            "importance": gating.get("importance", 0.0),
            "surprise": gating.get("surprise", 0.0),
            "emotion": gating.get("emotion", 0.0),
            "base_utility": gating.get("base_utility", 0.0),
            "relevance_score": gating.get("relevance_score", 0.0),
            "timestamp": datetime.utcnow().isoformat(),
            "frequency_count": 1
        }
    }
    
    return {
        "episodic_buffer": [memory_item], 
        "turn_count": state.get("turn_count", 0) + 1
    }


def consolidation_node(state: Dict[str, Any]):
    """
    The 'Sleep' Phase: 
    1. Filter high-utility memories.
    2. DEEP ENCODING: Extract entities using LLM (The 'Dream' phase).
    3. Save to Neo4j.
    """
    _ensure_llms()
    _ensure_neocortex()
    turn_count = state.get("turn_count", 0)
    buffer = state.get("episodic_buffer", [])
    
  
    if turn_count >= settings.CONSOLIDATION_INTERVAL and buffer:
        

        entity_counts = {}
        for m in buffer:
            for ent in m.get("entities", []):
                entity_counts[ent] = entity_counts.get(ent, 0) + 1

  
        high_utility_batch = []
        for m in buffer:
            base_utility = m.get("metadata", {}).get("base_utility", m.get("utility_score", 0.0))
            entities = m.get("entities", [])
            if entities:
                freq_count = max(entity_counts.get(ent, 1) for ent in entities)
            else:
                freq_count = 1
            freq_bonus = min((freq_count - 1) * settings.FREQUENCY_BONUS_STEP, settings.MAX_FREQUENCY_BONUS)
            replay_score = base_utility * (1.0 + freq_bonus)
            
            m.setdefault("metadata", {})
            m["metadata"]["frequency_count"] = freq_count
            m["metadata"]["replay_score"] = replay_score
            m["utility_score"] = replay_score

            if replay_score >= settings.MIN_CONSOLIDATION_SCORE and not m.get("is_noise", True):
                high_utility_batch.append(m)
        
        if high_utility_batch:
            print(f"--- SLEEP PHASE: Processing {len(high_utility_batch)} memories ---")
            
            for memory in high_utility_batch:
                try:
                    result_raw = entity_relation_chain.invoke({"input": memory["content"]})
                    result = normalize_entity_relation(result_raw)
                    entities = result.get("entities", [])
                    relations = result.get("relations", [])
                except Exception as e:
                    try:
                        raw_chain = ENTITY_RELATION_EXTRACTION_PROMPT | llm_structured | StrOutputParser()
                        raw = raw_chain.invoke({"input": memory["content"]})
                        parsed = parse_json_response(raw)
                        normalized = normalize_entity_relation(parsed)
                        entities = normalized.get("entities", []) or []
                        relations = normalized.get("relations", []) or []
                    except Exception as inner_e:
                        print(f" Entity-relation parsing failed: {inner_e}")
                        entities = []
                        relations = []

                try:
                    cleaned_entities = []
                    name_to_type = {}
                    allowed_types = {
                        "Project", "Issue", "Task", "Person", "Tool",
                        "Feature", "Location", "Date", "Other"
                    }
                    for ent in entities:
                        if not isinstance(ent, dict):
                            continue
                        name = str(ent.get("name", "")).strip()
                        etype = str(ent.get("type", "Other")).strip()
                        if not name:
                            continue
                        if etype not in allowed_types:
                            etype = "Other"
                        cleaned_entities.append({"name": name, "type": etype})
                        name_to_type[name] = etype


                    for rel in relations:
                        if not isinstance(rel, dict):
                            continue
                        for key in ["source", "target"]:
                            name = str(rel.get(key, "")).strip()
                            if name and name not in name_to_type:
                                cleaned_entities.append({"name": name, "type": "Other"})
                                name_to_type[name] = "Other"

                    cleaned_relations = []
                    for rel in relations:
                        if not isinstance(rel, dict):
                            continue
                        source = str(rel.get("source", "")).strip()
                        target = str(rel.get("target", "")).strip()
                        relation = str(rel.get("relation", "")).strip()
                        if not source or not target or not relation:
                            continue
                        if source not in name_to_type or target not in name_to_type:

                            continue
                        cleaned_relations.append({
                            "source": source,
                            "target": target,
                            "relation": relation,
                            "source_type": name_to_type[source],
                            "target_type": name_to_type[target]
                        })

                    memory["entities"] = cleaned_entities
                    memory["relations"] = cleaned_relations
                    print(f" Extracted Entities: {cleaned_entities}")
                    if cleaned_relations:
                        print(f" Extracted Relations: {cleaned_relations}")
                except Exception as e:
                    print(f"Extraction Failed for item: {e}")

                    try:
                        extraction_chain = ENTITY_EXTRACTION_PROMPT | llm_structured | StrOutputParser()
                        response = extraction_chain.invoke({"input": memory["content"]})
                        if "None" in response:
                            entities = []
                        else:
                            entities = [e.strip() for e in response.split(",") if e.strip()]
                        memory["entities"] = [{"name": e, "type": "Other"} for e in entities]
                        memory["relations"] = []
                    except Exception:
                        memory["entities"] = []
                        memory["relations"] = []

           
            try:
               
                neocortex.consolidate_batch(high_utility_batch)
                print(f"Saved batch to Graph.")
            except Exception as e:
                print(f"Save Error: {e}")

        
        return {"turn_count": 0, "episodic_buffer": "CLEAR"}
            
    return {}
