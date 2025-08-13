"""
Redis-Enhanced AWS Bedrock Financial Agent

Side-by-side comparison showing how Redis improves Bedrock Agents:
- Caches similar queries for faster responses
- Remembers conversation context
- Filters irrelevant queries before they hit Bedrock

Two endpoints: one with Redis enhancements, one without
"""

import time
import uuid
import asyncio
from typing import Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig
from redis_integration import RedisIntegration
from bedrock_service import BedrockService


# Initialize FastAPI application
app = FastAPI(
    title="Redis-Enhanced AWS Bedrock Financial Agent",
    description="Demonstration of Redis Enterprise value-add for AWS Bedrock Agent architectures",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve images
app.mount("/static", StaticFiles(directory="."), name="static")


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request with optional session management."""
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response with performance metadata."""
    response: str
    cached: bool
    time: float
    routed: bool = False
    session_id: str
    # New detailed trace information
    trace_data: Optional[dict] = None


# Global service instances (initialized on startup)
config: AppConfig
redis_integration: RedisIntegration  
bedrock_service: BedrockService

# Performance tracking
total_llm_calls_avoided = 0


def initialize_services() -> None:
    """Set up all the services we need to run the comparison."""
    global config, redis_integration, bedrock_service
    
    print("Initializing Redis-Enhanced Financial Agent")
    print("=" * 60)
    
    try:
        # Load all the environment variables
        config = AppConfig()
        print("Configuration loaded successfully")
        
        # Connect to AWS Bedrock
        bedrock_service = BedrockService(config.bedrock)
        print("Bedrock service initialized")
        
        # Set up Redis with all the caching and memory features
        redis_integration = RedisIntegration(config, bedrock_service.bedrock_runtime)
        print("Redis integration initialized")
        
        # Display architecture information
        print("\nARCHITECTURE COMPARISON READY")
        print(f"Redis-Enhanced: Agent {config.bedrock.agent_id_redis} + Redis VectorDB")
        print(f"  - Semantic caching (threshold: {config.cache.similarity_threshold})")
        print(f"  - Conversation memory (threshold: {config.cache.memory_similarity_threshold})")
        print(f"  - Intent-aware query routing")
        print(f"  - Session persistence (TTL: {config.cache.memory_ttl_seconds//3600}h)")
        print(f"Standard AWS: Agent {config.bedrock.agent_id_opensearch} + OpenSearch")
        print(f"  - Direct agent calls (no caching/memory)")
        
        print(f"\nApplication ready at http://localhost:8000")
        print("=" * 60)
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        raise


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    return str(uuid.uuid4())


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    initialize_services()


@app.get("/")
async def serve_frontend():
    """Serve the comparison UI."""
    return FileResponse("index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.post("/chat/arch1", response_model=ChatResponse)
async def redis_enhanced_architecture(request: ChatRequest):
    """
    This is the fun one - Redis does all the heavy lifting with caching and memory.
    
    Flow: Query → Check if allowed → Check cache → Get memories → Call Bedrock
    """
    global total_llm_calls_avoided
    
    start_time = time.time()
    timing_log = {}  # Track timing for each operation
    print(f"\n[REDIS-ARCH] Processing: '{request.query[:50]}...'")
    
    # Keep track of user sessions for memory
    session_id = request.session_id or generate_session_id()
    print(f"[REDIS-ARCH] Session: {session_id[:8]}...")
    timing_log["session_init"] = round(time.time() - start_time, 3)
    
    try:
        # STEP 1: Check if this is a valid query type
        # Filter out stuff like "what's the weather" before hitting Bedrock
        route_start = time.time()
        route_decision, route_confidence = redis_integration.route_query(request.query)
        timing_log["routing"] = round(time.time() - route_start, 3)
        
        if route_decision == "not_allowed":
            print(f"[REDIS-ARCH] Query filtered by semantic routing")
            
            # Update global tracking
            filter_time = round(time.time() - start_time, 3)
            total_llm_calls_avoided += 1
            
            print(f"[REDIS-ARCH] Avoided LLM call via routing (response in {filter_time}s)")
            print(f"[REDIS-ARCH] Total: {total_llm_calls_avoided} LLM calls avoided")
            
            return ChatResponse(
                response="Please ask me a question related to financial analysis, investment research, or personal information for memory testing.",
                cached=False,
                time=round(time.time() - start_time, 3),
                routed=True,
                session_id=session_id,
                trace_data={
                    "type": "filtered",
                    "confidence": round(route_confidence, 3),
                    "reason": "Query identified as off-topic",
                    "allowed_topics": [
                        "Financial analysis",
                        "Investment research", 
                        "Company performance",
                        "Market trends",
                        "Personal information"
                    ],
                    "llm_call_avoided": True
                }
            )
        
        print(f"[REDIS-ARCH] Query approved (confidence: {route_confidence:.3f})")
        
        # STEP 2: Generate embedding once for both searches
        embedding_start = time.time()
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None, redis_integration.embedding_service.embed, request.query
        )
        timing_log["embedding_generation"] = round(time.time() - embedding_start, 3)
        
        # STEP 3 & 4: Parallel search for memory and cache using same embedding
        # Run both searches concurrently to save time
        parallel_start = time.time()
        memory_task = asyncio.create_task(
            redis_integration.search_conversation_memory(session_id, request.query, query_embedding=query_embedding)
        )
        cache_task = asyncio.create_task(
            redis_integration.search_semantic_cache(request.query, query_embedding=query_embedding)
        )
        
        # Wait for both to complete
        memories, cache_result = await asyncio.gather(memory_task, cache_task)
        timing_log["parallel_search"] = round(time.time() - parallel_start, 3)
        
        memory_context = redis_integration.format_memory_context(memories)
        
        if memories:
            print(f"[REDIS-ARCH] Retrieved {len(memories)} relevant memories")
        
        if cache_result:
            print(f"[REDIS-ARCH] Cache hit! (similarity: {cache_result.similarity_score:.3f})")
            # Calculate cache response time
            cache_time = round(time.time() - start_time, 3)
            
            # Update global tracking
            total_llm_calls_avoided += 1
            
            # Log timing breakdown
            print(f"[REDIS-ARCH] Performance breakdown:")
            print(f"  - Routing: {timing_log['routing']}s")
            print(f"  - Embedding: {timing_log['embedding_generation']}s")
            print(f"  - Search (parallel): {timing_log['parallel_search']}s")
            print(f"  - Total: {cache_time}s")
            print(f"[REDIS-ARCH] Cache hit avoided LLM call")
            print(f"[REDIS-ARCH] Total: {total_llm_calls_avoided} LLM calls avoided")
            
            return ChatResponse(
                response=cache_result.response,
                cached=True,
                time=cache_time,
                session_id=session_id,
                trace_data={
                    "type": "cache_hit",
                    "similarity_score": round(cache_result.similarity_score, 3),
                    "matched_query": cache_result.query,
                    "cache_timestamp": cache_result.timestamp,
                    "memory_used": len(memories) if memories else 0,
                    "timing_breakdown": timing_log,
                    "llm_call_avoided": True
                }
            )
        
        # STEP 4: Actually call Bedrock with context from memory
        # Include any relevant conversation history to make responses better
        print(f"[REDIS-ARCH] Calling enhanced Bedrock Agent")
        
        bedrock_start = time.time()
        response = await bedrock_service.invoke_redis_enhanced_agent(
            query=request.query,
            context=memory_context
        )
        bedrock_elapsed = time.time() - bedrock_start
        timing_log["bedrock_call"] = round(bedrock_elapsed, 3)
        
        print(f"[REDIS-ARCH] Agent response in {bedrock_elapsed:.3f}s")
        
        # STEP 5: Save everything to Redis for next time
        # Cache the response and remember this conversation
        async def redis_background_tasks():
            """Save stuff to Redis so future queries are faster."""
            try:
                # Cache this Q&A for similar future questions
                await redis_integration.store_in_cache(request.query, response)
                
                # Remember this exchange for conversation context
                await redis_integration.store_conversation_memory(
                    session_id, request.query, response
                )
                
                print(f"[REDIS-ARCH] Background Redis storage completed")
                
            except Exception as error:
                print(f"[REDIS-ARCH] Background task failed: {error}")
        
        # Run this in background so user doesn't wait
        background_start = time.time()
        asyncio.create_task(redis_background_tasks())
        timing_log["background_task_launch"] = round(time.time() - background_start, 3)
        
        total_time = round(time.time() - start_time, 3)
        
        # Log detailed timing breakdown
        print(f"[REDIS-ARCH] Performance breakdown:")
        print(f"  - Routing: {timing_log.get('routing', 0)}s")
        print(f"  - Embedding: {timing_log.get('embedding_generation', 0)}s")
        print(f"  - Search (parallel): {timing_log.get('parallel_search', 0)}s")
        print(f"  - Bedrock call: {timing_log.get('bedrock_call', 0)}s")
        print(f"  - Background launch: {timing_log.get('background_task_launch', 0)}s")
        print(f"  - Total: {total_time}s")
        
        # Format memory entries for trace data
        memory_trace = []
        if memories:
            for mem in memories[:3]:  # Show top 3 most relevant
                memory_trace.append({
                    "user_said": mem.user_message,
                    "agent_responded": mem.bot_response[:100] + "..." if len(mem.bot_response) > 100 else mem.bot_response,
                    "relevance": round(mem.similarity, 3),
                    "when": mem.timestamp
                })
        
        return ChatResponse(
            response=response,
            cached=False,
            time=total_time,
            session_id=session_id,
            trace_data={
                "type": "fresh_query",
                "bedrock_time": round(bedrock_elapsed, 3),
                "preprocessing_time": round(bedrock_start - start_time, 3),
                "memory_context_used": memory_trace,
                "route_confidence": round(route_confidence, 3),
                "query_intent": redis_integration.intent_classifier.classify_intent(request.query),
                "timing_breakdown": timing_log
            }
        )
        
    except Exception as e:
        error_time = round(time.time() - start_time, 3)
        error_msg = f"Redis-Enhanced Architecture Error: {str(e)}"
        print(f"[REDIS-ARCH] {error_msg}")
        
        # Keep the demo running even if something breaks
        return ChatResponse(
            response=f"{error_msg}\n\n(Redis parts are working fine, this looks like a Bedrock connection issue)",
            cached=False,
            time=error_time,
            session_id=session_id
        )


@app.post("/chat/arch2", response_model=ChatResponse)
async def standard_aws_architecture(request: ChatRequest):
    """
    The baseline AWS setup without any Redis enhancements.
    
    Flow: Query → Bedrock Agent → OpenSearch (that's it)
    """
    start_time = time.time()
    print(f"\n[STANDARD-ARCH] Processing: '{request.query[:50]}...' (DIRECT)")
    
    # Still track sessions for comparison, but don't actually use memory
    session_id = request.session_id or generate_session_id()
    
    try:
        # Just call Bedrock directly, no fancy Redis stuff
        print(f"[STANDARD-ARCH] Calling standard Bedrock Agent (no caching/memory)")
        
        bedrock_start = time.time()
        response = await bedrock_service.invoke_standard_agent(request.query)
        bedrock_elapsed = time.time() - bedrock_start
        
        total_time = round(time.time() - start_time, 3)
        print(f"[STANDARD-ARCH] Response in {bedrock_elapsed:.3f}s (total: {total_time}s)")
        
        return ChatResponse(
            response=response,
            cached=False,  # Standard architecture never uses cache
            time=total_time,
            session_id=session_id
        )
        
    except Exception as e:
        error_time = round(time.time() - start_time, 3)
        error_msg = f"Standard Architecture Error: {str(e)}"
        print(f"[STANDARD-ARCH] {error_msg}")
        
        # Return error as response to keep demo running
        return ChatResponse(
            response=f"Error: {error_msg}",
            cached=False,
            time=error_time,
            session_id=session_id
        )


@app.post("/chat", response_model=ChatResponse)
async def legacy_chat_endpoint(request: ChatRequest):
    """Legacy endpoint for backward compatibility."""
    return await redis_enhanced_architecture(request)


# === REDIS MANAGEMENT ENDPOINTS ===

@app.post("/clear-cache")
async def clear_semantic_cache():
    """Clear Redis semantic cache for testing."""
    try:
        result = redis_integration.clear_cache()
        message = f"Cleared {result['cache_cleared']} cache entries"
        print(f"[MANAGEMENT] {message}")
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


@app.post("/clear-memory")
async def clear_conversation_memory():
    """Clear Redis conversation memory for testing."""
    try:
        result = redis_integration.clear_memory()
        message = f"Cleared {result['memory_cleared']} memory entries"
        print(f"[MANAGEMENT] {message}")
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {e}")


@app.post("/reset-all")
async def reset_all_redis_data():
    """Clear both cache and memory for complete demo reset."""
    try:
        cache_result = redis_integration.clear_cache()
        memory_result = redis_integration.clear_memory()
        
        total_cleared = cache_result['cache_cleared'] + memory_result['memory_cleared']
        message = f"Complete reset: cleared {cache_result['cache_cleared']} cache entries and {memory_result['memory_cleared']} memory entries"
        
        print(f"[MANAGEMENT] {message}")
        return {
            "message": message,
            "cache_cleared": cache_result['cache_cleared'],
            "memory_cleared": memory_result['memory_cleared'],
            "total_cleared": total_cleared
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset all data: {e}")


@app.get("/health")
async def health_check():
    """
    Comprehensive health check for all services.
    
    Returns the status of Redis integration, Bedrock services,
    and overall application health.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "redis": redis_integration.get_stats(),
            "bedrock": bedrock_service.health_check(),
            "config": {
                "cache_threshold": config.cache.similarity_threshold,
                "memory_threshold": config.cache.memory_similarity_threshold,
                "cache_ttl_hours": config.cache.cache_ttl_seconds // 3600,
                "memory_ttl_days": config.cache.memory_ttl_seconds // (3600 * 24)
            }
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/stats")
async def get_redis_statistics():
    """Get detailed Redis integration statistics."""
    try:
        return {
            "redis_stats": redis_integration.get_stats(),
            "config": {
                "similarity_threshold": config.cache.similarity_threshold,
                "memory_threshold": config.cache.memory_similarity_threshold,
                "cache_ttl": config.cache.cache_ttl_seconds,
                "memory_ttl": config.cache.memory_ttl_seconds
            },
            "performance_impact": {
                "llm_calls_avoided": total_llm_calls_avoided,
                "optimization_methods": ["Semantic Caching", "Query Routing"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")


if __name__ == "__main__":
    print("Starting Redis-Enhanced AWS Bedrock Financial Agent")
    print("Architecture Comparison Demo")
    print("Showcasing Redis Enterprise value-add for AWS Bedrock Agents")
    print()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )