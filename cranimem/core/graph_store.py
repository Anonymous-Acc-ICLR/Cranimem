# File: cranimem/core/graph_store.py
import hashlib
import json
import logging
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from neo4j import GraphDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import time
from cranimem.cognitive.prompts import ENTITY_RELATION_EXTRACTION_PROMPT, ENTITY_EXTRACTION_PROMPT
from cranimem.utils.json_utils import safe_json_load, parse_json_response, normalize_entity_relation
from neo4j.exceptions import ServiceUnavailable

logger = logging.getLogger(__name__)

class Neo4jNeocortex:
    """
    Long-term Semantic Memory (Neocortex).
    Implements HippoRAG-style retrieval with Associative Jump logic.
    """
    def __init__(self, uri, username, password, embedding_model):
        self.embedder = embedding_model
        self._warned_empty_rel = False
        for i in range(5):
            try:
                self.driver = GraphDatabase.driver(
                    uri,
                    auth=(username, password),
                    keep_alive=True,
                    connection_timeout=30,
                    max_connection_lifetime=3600,
                )
                self.driver.verify_connectivity()
                print("Successfully connected to Neo4j.")
                break
            except ServiceUnavailable:
                print(f" Neo4j not ready (Attempt {i+1}/5). Waiting...")
                time.sleep(5)
        else:
            raise Exception("Could not connect to Neo4j after 5 attempts.")
            
        self._initialize_schema()

    def close(self):
        """Closes the database connection."""
        self.driver.close()

    def query(self, cypher: str, **params):
        """Run an arbitrary Cypher query."""
        with self.driver.session() as session:
            return session.run(cypher, **params)

    def _initialize_schema(self):
        """Creates mandatory Vector Index and Entity constraints and waits for Consolidation state."""
        queries = [
            "DROP CONSTRAINT unique_memory_content IF EXISTS",
            "CREATE CONSTRAINT unique_memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            """
            CREATE VECTOR INDEX memory_embedding_index IF NOT EXISTS
            FOR (m:Memory) ON (m.embedding)
            OPTIONS {indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
            }}
            """
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Schema initialization warning: {e}")
            

            import time
            check_query = "SHOW INDEXES YIELD name, state WHERE name = 'memory_embedding_index' RETURN state"
            for i in range(10):  
                result = session.run(check_query).single()
                if result and result['state'] == 'ONLINE':
                    logger.info("Vector Index is ONLINE and ready.")
                    return
                logger.info(f"⏳ Waiting for vector index to populate... ({i+1}/10)")
                time.sleep(1)

    def _compute_hash(self, text: str) -> str:
        """MD5 Hash to prevent Neo4j index size errors on long text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def add_memory(self, text: str, importance: int, entities: list):
        """Robust Write using Hash ID."""
        if text is None:
            return
        mem_id = self._compute_hash(text)
        timestamp = datetime.utcnow().isoformat()
        embedding = self.embedder.embed_query(text)

        query = """
        MERGE (m:Memory {id: $mem_id})
        ON CREATE SET
            m.content = $text,
            m.importance = $importance,
            m.created_at = datetime($timestamp),
            m.last_accessed = datetime($timestamp),
            m.access_count = 1,
            m.embedding = $embedding
        ON MATCH SET
            m.last_accessed = datetime($timestamp),
            m.access_count = m.access_count + 1
        WITH m
        UNWIND $entities as entity_name
        MERGE (e:Entity {name: entity_name})
        MERGE (m)-[:RELATES_TO]->(e)
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    mem_id=mem_id,
                    text=text,
                    importance=importance,
                    timestamp=timestamp,
                    embedding=embedding,
                    entities=entities
                )
        except Exception as e:
            print(f" Graph Write Error: {e}")

    def consolidate_batch(self, buffer: List[Dict[str, Any]]):
        """
        Takes the raw Episodic Buffer and writes to Graph.
        Implements 'Repetition Count' by strengthening relationship weights.
        """
        if not buffer:
            return

        query = """
        UNWIND $batch AS item
        // 1. Create/Merge the Memory Node
        MERGE (m:Memory {id: item.id})
        ON CREATE SET 
            m.created_at = datetime(),
            m.access_count = 1,
            m.embedding = item.embedding,
            m.content = item.content
        ON MATCH SET 
            m.access_count = m.access_count + 1,
            m.last_accessed = datetime()

        // 2. Link to Entities with weight strengthening (Pattern Completion)
        FOREACH (entity IN item.entities |
            MERGE (e:Entity {name: entity.name})
            SET e.type = coalesce(entity.type, e.type)
            MERGE (m)-[r:RELATES_TO]->(e)
            ON CREATE SET r.weight = 1.0
            ON MATCH SET r.weight = r.weight + 0.2
        )

        // 3. Link Entity-to-Entity relations extracted by the LLM
        FOREACH (rel IN item.relations |
            MERGE (src:Entity {name: rel.source})
            SET src.type = coalesce(rel.source_type, src.type)
            MERGE (tgt:Entity {name: rel.target})
            SET tgt.type = coalesce(rel.target_type, tgt.type)
            MERGE (src)-[rr:RELATED_TO {type: rel.relation}]->(tgt)
        )
        """
        
        batch_data = []
        for item in buffer:
            emb = self.embedder.embed_query(item["content"])
            item["embedding"] = emb
            item["id"] = self._compute_hash(item["content"])

            entities = item.get("entities", []) or []
            normalized_entities = []
            for ent in entities:
                if isinstance(ent, dict):
                    name = str(ent.get("name", "")).strip()
                    etype = str(ent.get("type", "Other")).strip() if ent.get("type") else "Other"
                else:
                    name = str(ent).strip()
                    etype = "Other"
                if name:
                    normalized_entities.append({"name": name, "type": etype})
            item["entities"] = normalized_entities

            batch_data.append(item)

        if batch_data:
            with self.driver.session() as session:
                session.run(query, batch=batch_data)
                logger.info(f"Consolidated {len(batch_data)} episodic memories into Neo4j.")
                
                verify_result = session.run("""
                    MATCH (m:Memory)-[r:RELATES_TO]->(e:Entity)
                    RETURN count(DISTINCT m) as memories, count(DISTINCT e) as entities, count(r) as relationships
                """).single()
                logger.info(f"Graph now has: {verify_result['memories']} memories, {verify_result['entities']} entities, {verify_result['relationships']} relationships")

    def linkage_ingest(
        self,
        paragraphs: List[str],
        llm,
        batch_size: int = 5,
        max_retries: int = 2,
        max_chars: int = 2000,
        max_total_chars: int = 2000,
        verbose: bool = False
    ):
        """
        Builds a context graph by extracting entities/relations with an LLM
        and writing them to the KG via consolidate_batch.
        """
        if not paragraphs:
            return
        combined_text = "\n".join(paragraphs)
        if not combined_text.strip():
            return

        entities: List[str] = []

        try:
            prompt = (
                "Extract 5-10 important named entities (Person, Org, Location) "
                f"from: {combined_text[:1500]}. Return JSON list of strings."
            )
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                raw_list = json.loads(match.group(0))
                clean_entities = []
                for item in raw_list:
                    if isinstance(item, str):
                        clean_entities.append(item)
                    elif isinstance(item, dict) and "name" in item:
                        clean_entities.append(str(item["name"]))
                    elif isinstance(item, list):
                        clean_entities.extend([str(x) for x in item])
                entities = clean_entities
        except Exception:
            pass

        if not entities:
            if verbose:
                print("   -> LLM failed, using Regex fallback...")
            entities = list(set(re.findall(r'\b[A-Z][a-z0-9]+\b', combined_text)))[:10]

        self.add_memory(combined_text, importance=10, entities=entities)

    def retrieve_relevant(self, query: str, top_k: int = 3) -> List[str]:
        """Implements HippoRAG retrieval with fallback to simple vector search."""
        try:
            with self.driver.session() as session:
                rel_count = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
                if rel_count == 0:
                    if not self._warned_empty_rel:
                        logger.warning("⚠️ Retrieval skipped: No RELATES_TO relationships yet (graph is empty).")
                        self._warned_empty_rel = True
                    return []

            query_vector = self.embedder.embed_query(query)
            
            with self.driver.session() as session:
                rel_type_result = session.run("""
                    CALL db.relationshipTypes() YIELD relationshipType
                    RETURN count(CASE WHEN relationshipType = 'RELATED_TO' THEN 1 END) AS c
                """).single()
                has_related = (rel_type_result and rel_type_result["c"] > 0)

                if has_related:
                    hippo_query = """
                    CALL db.index.vector.queryNodes('memory_embedding_index', 10, $vector)
                    YIELD node AS seed, score AS sim
                    MATCH (seed)-[:RELATES_TO]->(e:Entity)
                    OPTIONAL MATCH (e)-[:RELATED_TO]-(related_entity:Entity)<-[:RELATES_TO]-(neighbor:Memory)
                    WHERE neighbor <> seed
                    WITH DISTINCT coalesce(neighbor, seed) as mem, sim
                    RETURN mem.content AS content, 
                        (sim * log(mem.access_count + 1)) AS score
                    ORDER BY score DESC LIMIT $top_k
                    """
                else:
                    hippo_query = """
                    CALL db.index.vector.queryNodes('memory_embedding_index', 10, $vector)
                    YIELD node AS seed, score AS sim
                    MATCH (seed)-[:RELATES_TO]->(e:Entity)
                    WITH DISTINCT seed as mem, sim
                    RETURN mem.content AS content, 
                        (sim * log(mem.access_count + 1)) AS score
                    ORDER BY score DESC LIMIT $top_k
                    """

                result = session.run(hippo_query, vector=query_vector, top_k=top_k)
                records = [record["content"] for record in result]
                
                if not records:
                    logger.debug("HippoRAG returned empty, falling back to direct vector search.")
                    fallback_query = """
                    CALL db.index.vector.queryNodes('memory_embedding_index', $top_k, $vector)
                    YIELD node, score
                    RETURN node.content AS content, score
                    ORDER BY score DESC
                    """
                    fallback_result = session.run(fallback_query, vector=query_vector, top_k=top_k)
                    records = [record["content"] for record in fallback_result]
                
                return records
                
        except Exception as e:
            if "no such vector schema index" in str(e).lower():
                logger.warning("Retrieval skipped: Vector index not yet ready or database is empty.")
            else:
                logger.error(f"Retrieval Error: {e}")
            return []
