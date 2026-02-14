# File: cranimem/cognitive/gating.py
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from..utils.llm_factory import get_llm
from..config import settings
from..utils.json_utils import safe_json_load

class TwoStageGating:
    def __init__(self, embedder: Embeddings):
        self.embedder = embedder
        self.llm = get_llm(temperature=0.0) 
        self.utility_parser = JsonOutputParser()
        
        self.utility_prompt = ChatPromptTemplate.from_template("""
        Analyze this input for long-term utility.
        Input: {input}
        Current Goal: {goal}
        
        IMPORTANT: Output ONLY raw JSON. Do not include any preamble, markdown, or explanations.
        Return JSON:
        {{
            "is_useful": boolean, 
            "entities": [list of strings], 
            "tool_calls": [list of strings]
        }}
        """)
        self.chain = self.utility_prompt | self.llm | self.utility_parser

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def evaluate(self, user_input: str, current_goal: str) -> dict:
        input_vec = self.embedder.embed_query(user_input)
        goal_vec = self.embedder.embed_query(current_goal)
        relevance_score = self.cosine_similarity(input_vec, goal_vec)
        
        if relevance_score < settings.RELEVANCE_THRESHOLD:
            return {
                "passed": False,
                "reason": "Low goal alignment",
                "relevance_score": relevance_score,
                "data": {}
            }

        try:
            analysis = self.chain.invoke({"input": user_input, "goal": current_goal})
            return {
                "passed": True,
                "relevance_score": relevance_score,
                "data": analysis 
            }
        except Exception:
            try:
                raw_chain = self.utility_prompt | self.llm | StrOutputParser()
                raw = raw_chain.invoke({"input": user_input, "goal": current_goal})
                parsed = safe_json_load(raw)
                if isinstance(parsed, dict):
                    return {
                        "passed": True,
                        "relevance_score": relevance_score,
                        "data": parsed
                    }
            except Exception:
                pass
            return {"passed": True, "relevance_score": relevance_score, "data": {}}