"""
AWS Bedrock Agent service functions.
"""

import json
import time
import asyncio
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BedrockConfig


class BedrockService:
    """
    Professional AWS Bedrock Agent integration.
    
    """
    
    def __init__(self, config: BedrockConfig):
        self.config = config
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.region)
        self.bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=config.region)
        
        print(f"[BEDROCK] Service initialized for region: {config.region}")
        print(f"[BEDROCK] Redis-enhanced agent: {config.agent_id_redis}")
        print(f"[BEDROCK] Standard agent: {config.agent_id_opensearch}")
    
    async def invoke_redis_enhanced_agent(self, query: str, context: str = "") -> str:
        """
        Invoke the Redis-enhanced Bedrock Agent.
        
        This agent has access to Redis Vector Database for enhanced
        knowledge retrieval and caching capabilities.
        
        Args:
            query: User query
            context: Additional context from conversation memory
            
        Returns:
            Agent response text
        """
        return await self._invoke_agent(
            agent_id=self.config.agent_id_redis,
            query=query,
            context=context,
            session_prefix="redis-arch"
        )
    
    async def invoke_standard_agent(self, query: str) -> str:
        """
        Invoke the standard AWS Bedrock Agent.
        
        This agent uses OpenSearch for knowledge retrieval,
        representing the baseline AWS architecture.
        
        Args:
            query: User query
            
        Returns:
            Agent response text
        """
        return await self._invoke_agent(
            agent_id=self.config.agent_id_opensearch,
            query=query,
            context="",
            session_prefix="standard-arch"
        )
    
    async def _invoke_agent(self, agent_id: str, query: str, 
                          context: str, session_prefix: str) -> str:
        """
        Core agent invocation with comprehensive error handling.
        
        Args:
            agent_id: Bedrock Agent ID
            query: User query
            context: Additional context to include
            session_prefix: Session identifier prefix
            
        Returns:
            Complete agent response
            
        Raises:
            Exception: For unrecoverable agent errors
        """
        start_time = time.time()
        
        try:
            # Prepare enhanced query with context
            enhanced_query = self._prepare_query(query, context)
            session_id = f"session-{session_prefix}-{int(time.time())}"
            
            print(f"[BEDROCK] Invoking agent {agent_id}")
            print(f"[BEDROCK] Query length: {len(enhanced_query)} characters")
            print(f"[BEDROCK] Session: {session_id}")
            
            # Run agent invocation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self._invoke_agent_sync,
                agent_id, 
                enhanced_query, 
                session_id
            )
            
            elapsed = time.time() - start_time
            print(f"[BEDROCK] Response received in {elapsed:.3f}s")
            print(f"[BEDROCK] Response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Agent {agent_id} failed after {elapsed:.3f}s: {str(e)}"
            print(f"[BEDROCK] ERROR: {error_msg}")
            print(f"[BEDROCK] Error type: {type(e).__name__}")
            raise Exception(error_msg)
    
    def _prepare_query(self, query: str, context: str) -> str:
        """
        Prepare enhanced query with conversation context.
        
        Args:
            query: Original user query
            context: Conversation context from memory
            
        Returns:
            Enhanced query string
        """
        if not context:
            return query
        
        return f"{context}\n\nCurrent question: {query}"
    
    def _invoke_agent_sync(self, agent_id: str, query: str, session_id: str) -> str:
        """
        Synchronous agent invocation for thread pool execution.
        
        Args:
            agent_id: Bedrock Agent ID
            query: Enhanced query text
            session_id: Unique session identifier
            
        Returns:
            Complete agent response text
            
        Raises:
            Various Bedrock-related exceptions
        """
        try:
            response = self.bedrock_agent_runtime.invoke_agent(
                agentId=agent_id,
                agentAliasId=self.config.agent_alias_id,
                sessionId=session_id,
                inputText=query
            )
            
            return self._process_streaming_response(response)
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise Exception(f"Bedrock ClientError [{error_code}]: {error_message}")
            
        except BotoCoreError as e:
            raise Exception(f"Bedrock BotoCoreError: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Unexpected Bedrock error: {str(e)}")
    
    def _process_streaming_response(self, response: dict) -> str:
        """
        Process Bedrock Agent streaming response.
        
        Args:
            response: Raw Bedrock response with event stream
            
        Returns:
            Complete response text
            
        Raises:
            Exception: If response processing fails
        """
        try:
            event_stream = response.get("completion")
            if not event_stream:
                raise Exception("No completion stream in Bedrock response")
            
            full_response = ""
            chunk_count = 0
            
            for event in event_stream:
                if "chunk" in event:
                    chunk = event["chunk"]["bytes"].decode("utf-8")
                    full_response += chunk
                    chunk_count += 1
                    
                elif "trace" in event:
                    # Log trace information for debugging
                    trace = event.get("trace", {})
                    trace_type = trace.get("type", "unknown")
                    if trace_type != "guardrailTrace":  # Reduce noise
                        print(f"[BEDROCK] Trace: {trace_type}")
            
            if not full_response:
                raise Exception("Empty response from Bedrock agent")
            
            print(f"[BEDROCK] Processed {chunk_count} response chunks")
            return full_response.strip()
            
        except Exception as e:
            raise Exception(f"Response processing failed: {str(e)}")
    
    def generate_embedding(self, text: str) -> list:
        """
        Generate embedding using Amazon Titan Embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            1536-dimensional embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.config.embedding_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text})
            )
            
            response_body = json.loads(response["body"].read())
            embedding = response_body["embedding"]
            
            if len(embedding) != self.config.embedding_dimensions:
                raise Exception(
                    f"Unexpected embedding dimensions: {len(embedding)} "
                    f"(expected {self.config.embedding_dimensions})"
                )
            
            return embedding
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            raise Exception(f"Embedding generation failed [{error_code}]: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Embedding error: {str(e)}")
    
    def health_check(self) -> dict:
        """
        Perform health check on Bedrock services.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            "bedrock_runtime": False,
            "bedrock_agent_runtime": False,
            "embedding_service": False
        }
        
        # Test Bedrock Runtime (embeddings)
        try:
            self.generate_embedding("test")
            health_status["bedrock_runtime"] = True
            health_status["embedding_service"] = True
        except Exception as e:
            print(f"[BEDROCK] Runtime health check failed: {e}")
        
        # Test Agent Runtime connectivity (without full invocation)
        try:
            # This is a minimal check - just verify the client is configured
            self.bedrock_agent_runtime.meta.client
            health_status["bedrock_agent_runtime"] = True
        except Exception as e:
            print(f"[BEDROCK] Agent runtime health check failed: {e}")
        
        return health_status