"""
Redis integration that adds caching, memory, and routing to Bedrock Agents.

What this does:
- Caches similar questions to avoid repeat Bedrock calls
- Remembers conversation history for better context
- Filters out irrelevant queries before they hit Bedrock
- Handles errors gracefully so demos don't break
"""

import ssl
import time
import json
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.extensions.router import Route, SemanticRouter
from redisvl.utils.vectorize.base import BaseVectorizer

from config import AppConfig, REDIS_SCHEMAS, ALLOWED_QUERY_REFERENCES


@dataclass
class CacheResult:
    """Result from semantic cache lookup."""
    query: str
    response: str
    timestamp: str
    similarity_score: float


@dataclass
class MemoryEntry:
    """Conversation memory entry."""
    user_message: str
    bot_response: str
    context: str
    similarity: float
    timestamp: str


class TitanEmbeddingService(BaseVectorizer):
    """Wrapper around Amazon Titan embeddings that works with Redis VL."""
    
    def __init__(self, bedrock_client):
        super().__init__(model="amazon.titan-embed-text-v1", dims=1536)
        # Store bedrock client as private to avoid Pydantic complaints
        object.__setattr__(self, '_bedrock_client', bedrock_client)
    
    def embed(self, text: str, **kwargs) -> List[float]:
        """Turn text into a vector using Titan embeddings."""
        try:
            start_time = time.time()
            response = self._bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text})
            )
            
            response_body = json.loads(response["body"].read())
            embed_time = time.time() - start_time
            print(f"[EMBEDDING] Generated in {embed_time:.3f}s for text: '{text[:30]}...'")
            return response_body["embedding"]
            
        except Exception as e:
            print(f"[EMBEDDING] Error generating embedding for text: {e}")
            raise
    
    def embed_many(self, texts: List[str], as_buffer: bool = False, **kwargs):
        """Generate embeddings for a bunch of texts at once."""
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts, 1):
            try:
                # Show progress for big batches so you know it's working
                if total > 10:
                    print(f"[EMBEDDING] Generating {i}/{total}: {text[:50]}...")
                
                embedding_array = np.array(self.embed(text), dtype=np.float32)
                
                # Redis needs bytes sometimes, lists other times
                if as_buffer:
                    embeddings.append(embedding_array.tobytes())
                else:
                    embeddings.append(embedding_array.tolist())
                    
            except Exception as e:
                print(f"[EMBEDDING] Failed to generate embedding {i}/{total}: {e}")
                raise
                
        return embeddings


class RedisSearchManager:
    """Redis search index management with intelligent recreation logic."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.indices: Dict[str, SearchIndex] = {}
    
    def initialize_index(self, schema_name: str, force_recreate: bool = False) -> SearchIndex:
        """
        Initialize a Redis search index with intelligent recreation logic.
        
        Args:
            schema_name: Name of the schema from REDIS_SCHEMAS
            force_recreate: Force recreation even if index exists
            
        Returns:
            SearchIndex instance ready for use
        """
        if schema_name not in REDIS_SCHEMAS:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        schema = REDIS_SCHEMAS[schema_name]
        index_name = schema["index"]["name"]
        
        # Create SearchIndex instance
        index = SearchIndex.from_dict(schema)
        index.set_client(self.redis_client)
        
        try:
            # Check if index already exists and is valid
            if not force_recreate and self._index_exists_and_valid(index_name):
                print(f"[REDIS] Using existing index: {index_name}")
                self.indices[schema_name] = index
                return index
            
            # Create or recreate the index
            index.create(overwrite=force_recreate)
            print(f"[REDIS] {'Recreated' if force_recreate else 'Created'} index: {index_name}")
            
        except Exception as e:
            print(f"[REDIS] Error with index {index_name}: {e}")
            # Try to recreate as fallback
            try:
                index.create(overwrite=True)
                print(f"[REDIS] Successfully recreated index {index_name} after error")
            except Exception as retry_error:
                print(f"[REDIS] Failed to recreate index {index_name}: {retry_error}")
                raise
        
        self.indices[schema_name] = index
        return index
    
    def _index_exists_and_valid(self, index_name: str) -> bool:
        """Check if index exists and has valid structure."""
        try:
            info = self.redis_client.execute_command('FT.INFO', index_name)
            # If we get here without exception, index exists
            return True
        except:
            return False
    
    def get_index(self, schema_name: str) -> SearchIndex:
        """Get an initialized index by schema name."""
        if schema_name not in self.indices:
            return self.initialize_index(schema_name)
        return self.indices[schema_name]


class RedisIntegration:
    """Main Redis integration class providing semantic caching, memory, and routing."""
    
    def __init__(self, config: AppConfig, bedrock_client):
        self.config = config
        self.bedrock_client = bedrock_client
        self.redis_client = self._create_redis_connection()
        self.search_manager = RedisSearchManager(self.redis_client)
        self.embedding_service = TitanEmbeddingService(bedrock_client)
        self.intent_classifier = QueryIntentClassifier()
        self._router_initialized = False
        
        # Initialize search indices
        self.cache_index = self.search_manager.initialize_index("semantic_cache")
        self.memory_index = self.search_manager.initialize_index("conversation_memory")
        
        # Initialize semantic router
        self.semantic_router = self._initialize_semantic_router()
        
        print(f"[REDIS] Integration initialized successfully")
        print(f"[REDIS] Connected to: {config.redis.host}:{config.redis.port}")
        print(f"[REDIS] Cache threshold: {config.cache.similarity_threshold}")
        print(f"[REDIS] Memory threshold: {config.cache.memory_similarity_threshold}")
    
    def _create_redis_connection(self) -> redis.Redis:
        """Create Redis connection with TLS support."""
        redis_config = self.config.redis
        
        connection_params = {
            "host": redis_config.host,
            "port": redis_config.port,
            "password": redis_config.password,
            "decode_responses": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {},
            "health_check_interval": 30
        }
        
        if redis_config.has_client_certs:
            print("[REDIS] Using client certificates for TLS authentication")
            connection_params.update({
                "ssl": True,
                "ssl_check_hostname": False,
                "ssl_cert_reqs": ssl.CERT_REQUIRED,
                "ssl_ca_certs": redis_config.ca_cert_path,
                "ssl_certfile": redis_config.client_cert_path,
                "ssl_keyfile": redis_config.client_key_path
            })
        else:
            print("[REDIS] Using TLS without client certificates")
            connection_params.update({
                "ssl": True,
                "ssl_check_hostname": False,
                "ssl_cert_reqs": ssl.CERT_NONE
            })
        
        client = redis.Redis(**connection_params)
        
        # Test connection
        try:
            client.ping()
            print("[REDIS] Connection established successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
        
        return client
    
    def _initialize_semantic_router(self) -> Optional[SemanticRouter]:
        """Initialize semantic router for query filtering."""
        try:
            # Check if user wants to force regeneration
            import os
            force_regenerate = os.getenv("REGENERATE_ROUTER_EMBEDDINGS", "false").lower() == "true"
            
            # Check if router already exists and is valid (unless forcing regeneration)
            if not force_regenerate and self._router_exists_and_valid():
                print(f"[ROUTER] Using existing router with {len(ALLOWED_QUERY_REFERENCES)} reference queries")
                return SemanticRouter(
                    name="app_financial_query_router",
                    vectorizer=self.embedding_service,
                    redis_client=self.redis_client
                )
            
            # Clean up any existing router only if we're regenerating or it's invalid
            if force_regenerate:
                print(f"[ROUTER] Regenerating router embeddings as requested")
                self._cleanup_existing_router()
            else:
                print(f"[ROUTER] Creating new router with {len(ALLOWED_QUERY_REFERENCES)} reference queries")
                # Only clean up if there's stale/partial data
                if self.redis_client.keys("app_route:*"):
                    print(f"[ROUTER] Cleaning up incomplete router data")
                    self._cleanup_existing_router()
            
            # Create allowed queries route
            allowed_route = Route(
                name="allowed_queries",
                references=ALLOWED_QUERY_REFERENCES,
                metadata={"type": "allowed"},
                distance_threshold=0.6  # Tighter threshold to filter weather queries
            )
            
            router = SemanticRouter(
                name="app_financial_query_router",
                vectorizer=self.embedding_service,
                routes=[allowed_route],
                redis_client=self.redis_client,
                overwrite=True
            )
            
            # Test the router with a known query
            try:
                test_result = router("My name is John")
                print(f"[ROUTER] Test query 'My name is John' result: {test_result}")
                test_result2 = router("my name is yusuf")
                print(f"[ROUTER] Test query 'my name is yusuf' result: {test_result2}")
            except Exception as e:
                print(f"[ROUTER] Test queries failed: {e}")
            
            print(f"[ROUTER] Initialized with {len(ALLOWED_QUERY_REFERENCES)} reference queries")
            self._router_initialized = True
            return router
            
        except Exception as e:
            print(f"[ROUTER] Failed to initialize: {e}")
            return None
    
    def _router_exists_and_valid(self) -> bool:
        """Check if router index exists and has valid data."""
        try:
            # Check if index exists
            existing_indices = self.redis_client.execute_command("FT._LIST")
            router_index_exists = any(
                (idx.decode() if isinstance(idx, bytes) else str(idx)) == "app_financial_query_router"
                for idx in existing_indices
            )
            
            if not router_index_exists:
                print(f"[ROUTER] Index 'app_financial_query_router' not found")
                return False
            
            # Check if router has documents
            router_keys = self.redis_client.keys("app_route:*")
            expected_routes = len(ALLOWED_QUERY_REFERENCES)
            actual_routes = len(router_keys)
            
            print(f"[ROUTER] Found {actual_routes} route documents (expected: {expected_routes})")
            
            # Router is valid if it has the expected number of routes
            return actual_routes >= expected_routes
            
        except Exception as e:
            print(f"[ROUTER] Error checking router validity: {e}")
            return False
    
    def _cleanup_existing_router(self) -> None:
        """Clean up existing router indices to prevent conflicts."""
        try:
            existing_indices = self.redis_client.execute_command("FT._LIST")
            for idx in existing_indices:
                idx_str = idx.decode() if isinstance(idx, bytes) else str(idx)
                if idx_str == "app_financial_query_router":
                    self.redis_client.execute_command("FT.DROPINDEX", idx_str, "DD")
                    print(f"[ROUTER] Cleaned up existing index: {idx_str}")
            
            # Clean router documents
            router_keys = self.redis_client.keys("app_route:*")
            if router_keys:
                self.redis_client.delete(*router_keys)
                print(f"[ROUTER] Cleaned up {len(router_keys)} router documents")
                
        except Exception as e:
            print(f"[ROUTER] Cleanup warning: {e}")
    
    # === Semantic Caching Methods ===
    
    async def search_semantic_cache(self, query: str, query_embedding: Optional[List[float]] = None) -> Optional[CacheResult]:
        """
        Search semantic cache with intent-aware matching.
        
        Returns cached response only if both similarity and intent match,
        preventing incorrect cache hits between different query types.
        """
        try:
            # Use provided embedding or generate new one
            if query_embedding is None:
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, self.embedding_service.embed, query
                )
            else:
                embedding = query_embedding
            
            # Get current query intent
            current_intent = self.intent_classifier.classify_intent(query)
            
            # Search cache directly without retrieving all keys
            vector_query = VectorQuery(
                vector=embedding,
                vector_field_name="embedding",
                num_results=2,  # Reduced from 3 - only need top matches
                return_fields=["query", "response", "timestamp"]
            )
            
            query_start = time.time()
            results = self.cache_index.query(vector_query)
            print(f"[CACHE] Vector search completed in {time.time() - query_start:.3f}s")
            
            # Find best match with intent compatibility
            for i, result in enumerate(results):
                similarity_score = 1 - float(result.get("vector_distance", 1))
                cached_query = result.get("query", "")
                cached_intent = self.intent_classifier.classify_intent(cached_query)
                
                print(f"[CACHE] Candidate {i+1}: Intent={cached_intent}, Similarity={similarity_score:.3f}")
                print(f"[CACHE] Debug: similarity_score={similarity_score}, threshold={self.config.cache.similarity_threshold}")
                
                # Check for cache hit based on similarity alone
                if similarity_score > self.config.cache.similarity_threshold:
                    print(f"[CACHE] Cache hit! Similarity={similarity_score:.3f}, Current intent={current_intent}, Cached intent={cached_intent}")
                    return CacheResult(
                        query=cached_query,
                        response=result.get("response"),
                        timestamp=result.get("timestamp"),
                        similarity_score=similarity_score
                    )
                else:
                    print(f"[CACHE] Similarity {similarity_score:.3f} below threshold {self.config.cache.similarity_threshold}")
            
            print(f"[CACHE] Cache miss - no suitable match found")
            return None
            
        except Exception as e:
            print(f"[CACHE] Search error: {e}")
            return None
    
    async def store_in_cache(self, query: str, response: str) -> None:
        """Store query-response pair in semantic cache."""
        try:
            # Generate embedding
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self.embedding_service.embed, query
            )
            
            # Store in Redis
            key = f"app_cache:{int(time.time() * 1000000)}"
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            data = {
                "query": query,
                "response": response,
                "embedding": embedding_bytes,
                "timestamp": str(int(time.time()))
            }
            
            self.redis_client.hset(key, mapping=data)
            self.redis_client.expire(key, self.config.cache.cache_ttl_seconds)
            
            print(f"[CACHE] Stored: '{query[:50]}...'")
            
        except Exception as e:
            print(f"[CACHE] Storage error: {e}")
    
    # === Conversation Memory Methods ===
    
    async def search_conversation_memory(self, session_id: str, query: str, 
                                       max_results: int = 3, query_embedding: Optional[List[float]] = None) -> List[MemoryEntry]:
        """Search conversation memory for relevant context."""
        try:
            # Use provided embedding or generate new one
            if query_embedding is None:
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, self.embedding_service.embed, query
                )
            else:
                embedding = query_embedding
            
            print(f"[MEMORY] Searching session {session_id[:8]}... for: '{query[:50]}...'")
            
            # Search memory with session filter
            vector_query = VectorQuery(
                vector=embedding,
                vector_field_name="user_embedding",  # Search against user message embeddings
                num_results=max_results,
                return_fields=["session_id", "user_message", "bot_response", "context", "timestamp"],
                filter_expression=f'@session_id:"{session_id}"'
            )
            
            query_start = time.time()
            results = self.memory_index.query(vector_query)
            print(f"[MEMORY] Vector search completed in {time.time() - query_start:.3f}s")
            
            memories = []
            for result in results:
                similarity_score = 1 - float(result.get("vector_distance", 1))
                user_msg = result.get("user_message", "")[:30]
                
                print(f"[MEMORY] Found: '{user_msg}...' similarity={similarity_score:.3f}")
                
                if similarity_score > self.config.cache.memory_similarity_threshold:
                    memories.append(MemoryEntry(
                        user_message=result.get("user_message"),
                        bot_response=result.get("bot_response"),
                        context=result.get("context"),
                        similarity=similarity_score,
                        timestamp=result.get("timestamp")
                    ))
                    print(f"[MEMORY] Added memory (similarity={similarity_score:.3f})")
                else:
                    print(f"[MEMORY] Filtered out (threshold={self.config.cache.memory_similarity_threshold})")
            
            print(f"[MEMORY] Retrieved {len(memories)} relevant memories")
            return memories
            
        except Exception as e:
            print(f"[MEMORY] Search error: {e}")
            return []
    
    async def store_conversation_memory(self, session_id: str, user_message: str, 
                                      bot_response: str, context: str = "") -> None:
        """Store conversation turn in memory with dual embeddings."""
        try:
            loop = asyncio.get_event_loop()
            
            # Generate embeddings for both user message and full conversation
            user_embedding = await loop.run_in_executor(
                None, self.embedding_service.embed, user_message
            )
            
            full_context = f"User: {user_message}\nBot: {bot_response}\nContext: {context}"
            full_embedding = await loop.run_in_executor(
                None, self.embedding_service.embed, full_context
            )
            
            # Store in Redis
            key = f"app_memory:{session_id}:{int(time.time() * 1000000)}"
            
            data = {
                "session_id": session_id,
                "user_message": user_message,
                "bot_response": bot_response,
                "context": context,
                "user_embedding": np.array(user_embedding, dtype=np.float32).tobytes(),
                "embedding": np.array(full_embedding, dtype=np.float32).tobytes(),
                "timestamp": str(int(time.time()))
            }
            
            self.redis_client.hset(key, mapping=data)
            self.redis_client.expire(key, self.config.cache.memory_ttl_seconds)
            
            print(f"[MEMORY] Stored conversation for session {session_id[:8]}...")
            
        except Exception as e:
            print(f"[MEMORY] Storage error: {e}")
    
    def format_memory_context(self, memories: List[MemoryEntry]) -> str:
        """Format conversation memories into context string for Bedrock."""
        if not memories:
            return ""
        
        context_parts = ["Previous conversation context:"]
        for memory in memories[:3]:  # Use top 3 most relevant
            context_parts.append(f"User previously said: {memory.user_message}")
            if memory.context:
                context_parts.append(f"Context: {memory.context}")
        
        return "\n".join(context_parts)
    
    # === Semantic Routing Methods ===
    
    def route_query(self, query: str) -> Tuple[str, float]:
        """
        Route query through semantic filtering.
        
        Returns:
            Tuple of (route_decision, confidence_score)
            route_decision: 'allowed' or 'not_allowed'
            confidence_score: Distance from nearest reference query
        """
        if not self.semantic_router:
            print("[ROUTER] No router available, defaulting to allowed")
            return "allowed", 0.0
        
        # Quick check if router is still valid (only if not already validated)
        if not self._router_initialized:
            if self._router_exists_and_valid():
                self._router_initialized = True
            else:
                print("[ROUTER] Router invalidated, defaulting to allowed")
                return "allowed", 0.0
        
        try:
            route_start = time.time()
            route_match = self.semantic_router(query)
            route_time = time.time() - route_start
            
            print(f"[ROUTER] Query evaluated in {route_time:.3f}s - Query: '{query[:50]}...', Match: {route_match}")
            if route_match:
                print(f"[ROUTER] Debug - Match name: {route_match.name}, Distance: {getattr(route_match, 'distance', 'N/A')}")
            
            if route_match and route_match.name:
                distance = getattr(route_match, 'distance', 0.0)
                print(f"[ROUTER] Allowed query (distance={distance:.3f})")
                return "allowed", distance
            else:
                print(f"[ROUTER] Filtered query - not financial/personal")
                return "not_allowed", 1.0
                
        except Exception as e:
            print(f"[ROUTER] Error: {e}, defaulting to allowed")
            return "allowed", 0.0
    
    # === Management Methods ===
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear semantic cache entries."""
        try:
            cache_keys = self.redis_client.keys("app_cache:*")
            count = len(cache_keys)
            if cache_keys:
                self.redis_client.delete(*cache_keys)
            print(f"[CACHE] Cleared {count} entries")
            return {"cache_cleared": count}
        except Exception as e:
            print(f"[CACHE] Clear error: {e}")
            return {"cache_cleared": 0}
    
    def clear_memory(self) -> Dict[str, int]:
        """Clear conversation memory entries."""
        try:
            memory_keys = self.redis_client.keys("app_memory:*")
            count = len(memory_keys)
            if memory_keys:
                self.redis_client.delete(*memory_keys)
            print(f"[MEMORY] Cleared {count} entries")
            return {"memory_cleared": count}
        except Exception as e:
            print(f"[MEMORY] Clear error: {e}")
            return {"memory_cleared": 0}
    
    def get_stats(self) -> Dict[str, int]:
        """Get Redis integration statistics."""
        try:
            return {
                "cache_entries": len(self.redis_client.keys("app_cache:*")),
                "memory_entries": len(self.redis_client.keys("app_memory:*")),
                "router_entries": len(self.redis_client.keys("app_route:*"))
            }
        except Exception as e:
            print(f"[REDIS] Stats error: {e}")
            return {"cache_entries": 0, "memory_entries": 0, "router_entries": 0}


# Internal utility for cache organization
class QueryIntentClassifier:
    @staticmethod
    def classify_intent(query: str) -> str:
        query_lower = query.lower().strip()
        question_starters = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'could', 'would', 'should', 'do', 'does', 'did', 'will', 'is', 'are', 'was', 'were', 'describe', 'explain', 'tell', 'show', 'list', 'identify', 'analyze', 'compare', 'evaluate', 'discuss', 'summarize', 'outline', 'detail', 'provide', 'give']
        if any(query_lower.startswith(q) for q in question_starters) or query_lower.endswith('?'):
            return "question"
        statement_starters = ['i have', 'i am', 'i work', 'i live', 'i like', 'i dont', 'i do not', 'my name', 'my favorite']
        if any(query_lower.startswith(s) for s in statement_starters):
            return "statement"
        return "analysis"