import argparse
import os
import random
import time
import json
import re
import numpy as np
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from datasets import load_dataset
from langchain_core.messages import HumanMessage, AIMessage


from memagents.config import settings
from memagents.core.embedding import get_embeddings
from memagents.core.graph_store import Neo4jNeocortex
from memagents.agent.workflow import build_agent
from memagents.agent.nodes import consolidation_node 



def normalize_text(s: str) -> str:
    """Standard normalization for F1/EM."""
    if not s: return ""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return re.sub(r"[^\w\s]", "", text)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_f1_score(prediction: str, ground_truth: str):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0, 0.0, 0.0
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0.0, 0.0, 0.0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1

def strip_chatty(text: str) -> str:
    """
    Robust cleanup:
    1) Extract <RESPONSE> tags if present (highest priority).
    2) Fallback to removing chatty prefixes/suffixes and boilerplate.
    """
    if not text:
        return ""
    match = re.search(r"<RESPONSE>(.*?)</RESPONSE>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    cleaned = text.strip()
    if "Assistant:" in cleaned:
        cleaned = cleaned.split("Assistant:")[-1].strip()
    patterns = [
        r"^here( is|'s) (your )?answer[:\s-]*",
        r"^the answer is[:\s-]*",
        r"^answer[:\s-]*",
        r"^based on the context[,:\s-]*",
        r"^according to (the )?context[,:\s-]*",
        r"^in summary[,:\s-]*",
    ]
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    # Remove trailing filler
    cleaned = re.sub(r"\s*(thanks|thank you|hope that helps|let me know).*?$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def llm_judge_correctness(question: str, prediction: str, ground_truth: str, llm) -> float:
    """
    Uses an LLM to decide if the answer is semantically correct.
    Returns 1.0 (Correct) or 0.0 (Incorrect).
    """
    prompt = f"""
    [Judge the Answer]
    Question: {question}
    Ground Truth: {ground_truth}
    Agent Answer: {prediction}

    Is the Agent Answer semantically correct based on the Ground Truth? 
    Ignore minor phrasing differences. 
    Respond with exactly one word: "YES" or "NO".
    """
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        return 1.0 if "YES" in text.upper() else 0.0
    except:
        return 0.0


def reset_agent_memory(neocortex):
    """Hard reset: wipe Neo4j and restart."""
    print("   [RESET] Wiping Knowledge Graph...")
    neocortex.query("MATCH (n) DETACH DELETE n")

def run_cranimem_test(
    question: str,
    answer: str,
    context_facts: List[str],
    distractor_facts: List[str],
    agent_executor,
    neocortex,
    llm,
    fast_ingest: bool = False
):
    """
    Executes the 'Torture Test':
    1. INGEST: Stream mixed signal/noise to the agent.
    2. CONSOLIDATE: Force memory consolidation.
    3. RETRIEVE: Ask the question.
    """

    study_goal = "Store all facts and details provided in the input into memory."
    

    raw_stream = context_facts + distractor_facts
    random.shuffle(raw_stream)
    
    if fast_ingest:
        print(f"   [Step 1] Rapidly ingesting {len(raw_stream)} facts in STUDY MODE (fast_ingest)...")
        
  
        simulated_buffer = [
            {
                "content": fact,
                "utility_score": 1.0,
                "is_noise": False,
                "entities": [],
                "metadata": {
                    "importance": 1.0,
                    "surprise": 0.0,
                    "emotion": 0.0,
                    "base_utility": 1.0,
                    "relevance_score": 1.0,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "frequency_count": 1
                }
            }
            for fact in raw_stream
        ]
        
        state = {
            "messages": [],
            "episodic_buffer": simulated_buffer,
            "current_goal": study_goal,
            "turn_count": len(simulated_buffer)
        }
    else:
        print(f"   [Step 1] Streaming {len(raw_stream)} inputs in STUDY MODE...")
        state = {
            "messages": [],
            "episodic_buffer": [],
            "current_goal": study_goal,
            "turn_count": 0
        }
        for i, fact in enumerate(raw_stream):
            print(f"      Processing item {i+1}/{len(raw_stream)}...", end="\r")
            state["current_input"] = fact
            result = agent_executor.invoke(state)
            state["episodic_buffer"] = result.get("episodic_buffer", [])
            state["turn_count"] += 1

    print(f"   [Step 2] Forcing Consolidation (Buffer Size: {len(state['episodic_buffer'])})...")
    

    final_state = consolidation_node(state)
    

    print("   [Step 3] Asking the Question in EXAM MODE...")
    

    exam_state = {
        "messages": [],
        "episodic_buffer": [], 
        "current_goal": question,
        "current_input": question,
        "turn_count": 0
    }
    

    exam_result = agent_executor.invoke(exam_state)
    prediction = strip_chatty(exam_result.get("agent_response", ""))
    

    prec, rec, f1 = compute_f1_score(prediction, answer)
    semantic_score = llm_judge_correctness(question, prediction, answer, llm)
    
    return {
        "prediction": prediction,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "semantic_accuracy": semantic_score
    }

def run_rag_test(
    question: str,
    answer: str,
    context_facts: List[str],
    distractor_facts: List[str],
    llm,
    embedder,
    top_k: int = 3,
):
    """
    Simple RAG baseline:
    1) Retrieve top-k facts by embedding similarity from context + distractors
    2) Answer with a short, non-chatty response (2-3 words).
    """
    pool = context_facts + distractor_facts
    if not pool:
        prediction = ""
    else:
        q_vec = embedder.embed_query(question)
        doc_vecs = embedder.embed_documents(pool)
        # cosine similarity
        sims = []
        q_norm = np.linalg.norm(q_vec) + 1e-8
        for i, v in enumerate(doc_vecs):
            v_norm = np.linalg.norm(v) + 1e-8
            sims.append((float(np.dot(q_vec, v) / (q_norm * v_norm)), i))
        sims.sort(reverse=True, key=lambda x: x[0])
        top = [pool[i] for _, i in sims[:max(1, top_k)]]
        rag_prompt = f"""
You are a precise answering machine.
Use ONLY the context to answer the question.

RULES:
1. Output ONLY the answer (Entity, Name, Date, or Place).
2. Do NOT write a sentence.
3. Do NOT use "The answer is".

Context:
{chr(10).join(top)}

Question:
{question}

Answer:
"""
        response = llm.invoke(rag_prompt)
        prediction = response.content if hasattr(response, "content") else str(response)
    prediction = strip_chatty(prediction)
    prec, rec, f1 = compute_f1_score(prediction, answer)
    semantic_score = llm_judge_correctness(question, prediction, answer, llm)
    return {
        "prediction": prediction,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "semantic_accuracy": semantic_score
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization for local_hf provider")
    parser.add_argument("--csv", type=str, default=None, help="Write per-question results to CSV file")
    parser.add_argument("--fast_ingest", action="store_true", help="Skip gating during study to speed up ingestion")
    parser.add_argument("--rag_top_k", type=int, default=3, help="Top-k facts for RAG baseline retrieval")
    args = parser.parse_args()
    ENABLE_RAG = False


    print(">>> Initializing Cranimem Eval...")
    embeddings = get_embeddings()
    

    from memagents.utils.llm_factory import get_llm
    llm = get_llm(temperature=0.1, use_4bit=args.use_4bit)
    
    neocortex = Neo4jNeocortex(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD.get_secret_value(),
        embedding_model=embeddings
    )
    
    agent = build_agent()
    

    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{args.samples}]")
    
    metrics = {
        "f1": [], "prec": [], "rec": [], "semantic": []
    }
    rag_metrics = {
        "f1": [], "prec": [], "rec": [], "semantic": []
    }
    mem_latency = []
    rag_latency = []
    mem_predictions = []
    rag_predictions = []
    references = []
    csv_rows = []

    print(f"\n>>> Starting Torture Test on {args.samples} Questions <<<\n")

    for i, row in enumerate(ds):
        question = row['question']
        answer = row['answer']
        

        context = row['context'] 
        supporting_facts = [] 
        for item in context:
            text = "".join(item[1])
            supporting_facts.append(text)
            

        noise = [
            "The sky is blue and the grass is green.",
            "Python is a programming language released in 1991.",
            "Pizza is a popular dish in Italy."
        ]
        
        print(f"--- Test {i+1}: {question} ---")
        

        reset_agent_memory(neocortex)
        

        mem_start = time.perf_counter()
        result = run_cranimem_test(
            question, answer, supporting_facts, noise, 
            agent, neocortex, llm, fast_ingest=args.fast_ingest
        )
        mem_elapsed = time.perf_counter() - mem_start
        if ENABLE_RAG:
            rag_start = time.perf_counter()
            rag_result = run_rag_test(
                question, answer, supporting_facts, noise,
                llm, embeddings, top_k=args.rag_top_k
            )
            rag_elapsed = time.perf_counter() - rag_start
        

        print(f"   [Result] Pred: {result['prediction'][:50]}... | Truth: {answer}")
        print(f"   [Scores] F1: {result['f1']:.2f} | Semantic: {result['semantic_accuracy']}")
        if ENABLE_RAG:
            print(f"   [RAG]    Pred: {rag_result['prediction'][:50]}... | F1: {rag_result['f1']:.2f} | Semantic: {rag_result['semantic_accuracy']}")
        
        metrics["f1"].append(result["f1"])
        metrics["prec"].append(result["precision"])
        metrics["rec"].append(result["recall"])
        metrics["semantic"].append(result["semantic_accuracy"])
        mem_latency.append(mem_elapsed)
        if ENABLE_RAG:
            rag_metrics["f1"].append(rag_result["f1"])
            rag_metrics["prec"].append(rag_result["precision"])
            rag_metrics["rec"].append(rag_result["recall"])
            rag_metrics["semantic"].append(rag_result["semantic_accuracy"])
            rag_latency.append(rag_elapsed)
        mem_predictions.append(result["prediction"])
        if ENABLE_RAG:
            rag_predictions.append(rag_result["prediction"])
        references.append(answer)
        if args.csv:
            row = {
                "idx": i + 1,
                "question": question,
                "answer": answer,
                "prediction": result["prediction"],
                "f1": result["f1"],
                "precision": result["precision"],
                "recall": result["recall"],
                "semantic_accuracy": result["semantic_accuracy"],
                "latency_s": mem_elapsed,
            }
            if ENABLE_RAG:
                row.update({
                    "rag_prediction": rag_result["prediction"],
                    "rag_f1": rag_result["f1"],
                    "rag_precision": rag_result["precision"],
                    "rag_recall": rag_result["recall"],
                    "rag_semantic_accuracy": rag_result["semantic_accuracy"],
                    "rag_latency_s": rag_elapsed,
                })
            csv_rows.append(row)


    avg_prec = np.mean(metrics["prec"]) if metrics["prec"] else 0.0
    avg_rec = np.mean(metrics["rec"]) if metrics["rec"] else 0.0
    avg_f1 = np.mean(metrics["f1"]) if metrics["f1"] else 0.0
    avg_sem = np.mean(metrics["semantic"]) if metrics["semantic"] else 0.0
    rag_avg_prec = np.mean(rag_metrics["prec"]) if rag_metrics["prec"] else 0.0
    rag_avg_rec = np.mean(rag_metrics["rec"]) if rag_metrics["rec"] else 0.0
    rag_avg_f1 = np.mean(rag_metrics["f1"]) if rag_metrics["f1"] else 0.0
    rag_avg_sem = np.mean(rag_metrics["semantic"]) if rag_metrics["semantic"] else 0.0
    avg_mem_latency = np.mean(mem_latency) if mem_latency else 0.0
    avg_rag_latency = np.mean(rag_latency) if rag_latency else 0.0

  

    print("\n" + "=" * 64)
    print("EVAL PERFORMANCE SUMMARY (MEMAGENTS)")
    print("=" * 64)
    if ENABLE_RAG:
        print(f"{'Metric':<18} | {'MemAgents':>10} | {'RAG':>10}")
    else:
        print(f"{'Metric':<18} | {'Score':>10}")
    print("-" * 64)
    if ENABLE_RAG:
        print(f"{'Precision':<18} | {avg_prec:>10.3f} | {rag_avg_prec:>10.3f}")
        print(f"{'Recall':<18} | {avg_rec:>10.3f} | {rag_avg_rec:>10.3f}")
        print(f"{'F1':<18} | {avg_f1:>10.3f} | {rag_avg_f1:>10.3f}")
        print(f"{'Semantic Acc':<18} | {avg_sem:>10.3f} | {rag_avg_sem:>10.3f}")
        print(f"{'Latency (s)':<18} | {avg_mem_latency:>10.3f} | {avg_rag_latency:>10.3f}")
    else:
        print(f"{'Precision':<18} | {avg_prec:>10.3f}")
        print(f"{'Recall':<18} | {avg_rec:>10.3f}")
        print(f"{'F1':<18} | {avg_f1:>10.3f}")
        print(f"{'Semantic Acc':<18} | {avg_sem:>10.3f}")
        print(f"{'Latency (s)':<18} | {avg_mem_latency:>10.3f}")
    print("=" * 64)
    
    if args.csv:
        try:
            import csv
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "idx",
                        "question",
                        "answer",
                        "prediction",
                        "f1",
                        "precision",
                        "recall",
                        "semantic_accuracy",
                        "latency_s",
                    ],
                )
                if ENABLE_RAG:
                    writer.fieldnames.extend([
                        "rag_prediction",
                        "rag_f1",
                        "rag_precision",
                        "rag_recall",
                        "rag_semantic_accuracy",
                        "rag_latency_s",
                    ])
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\n>>> Wrote CSV results to {args.csv} <<<")
        except Exception as e:
            print(f"\nFailed to write CSV: {e}")

if __name__ == "__main__":
    main()
