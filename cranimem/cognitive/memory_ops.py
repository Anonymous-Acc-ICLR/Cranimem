# File: cranimem/cognitive/memory_ops.py
import logging
from ..core.graph_store import Neo4jNeocortex
from ..core.embedding import get_embeddings
from ..config import settings

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    High-level facade for memory maintenance operations.
    Useful for background jobs or manual administration.
    """
    def __init__(self):
        self.embedder = get_embeddings()
        self.store = Neo4jNeocortex(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD.get_secret_value(),
            embedding_model=self.embedder
        )

    def force_consolidation(self):
        """
        Triggers the pruning process manually.
        This implements the 'Optimization (Consolidation)' phase.
        """
        logger.info("Starting manual memory consolidation...")
        self.store.prune_memories(utility_threshold=0.2, days_unused=180)
        logger.info("Consolidation complete.")

    def clear_short_term_buffer(self):
        """
        Example utility: Clear low-importance memories immediately 
        if the graph grows too large (Maintenance).
        """
        
        with self.store.driver.session() as session:
            count = session.run("MATCH (m:Memory) RETURN count(m) as c").single()['c']
            
        if count > 1000:
            logger.warning("Memory buffer high. Pruning only very low-utility, old memories...")
            self.store.prune_memories(utility_threshold=0.1, days_unused=365)

    def close(self):
        self.store.close()
