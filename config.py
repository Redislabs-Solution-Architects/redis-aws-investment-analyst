"""
This module centralizes all config settings, environment variables,
and application constants.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class RedisConfig:
    """Redis Enterprise connection configuration."""
    host: str
    port: int
    password: str
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None
    
    @property
    def has_client_certs(self) -> bool:
        """Check if client certificates are available for TLS authentication."""
        return all([self.ca_cert_path, self.client_cert_path, self.client_key_path])


@dataclass
class BedrockConfig:
    """AWS Bedrock service configuration."""
    region: str
    agent_id_redis: str  # Agent with Redis vector database
    agent_id_opensearch: str  # Agent with OpenSearch database
    agent_alias_id: str = "TSTALIASID"
    embedding_model: str = "amazon.titan-embed-text-v1"
    embedding_dimensions: int = 1536


@dataclass
class CacheConfig:
    """Semantic caching configuration."""
    similarity_threshold: float = 0.70  # Lowered to catch more similar queries
    cache_ttl_seconds: int = 86400  # 24 hours
    memory_ttl_seconds: int = 3600 * 24 * 7  # 7 days
    memory_similarity_threshold: float = 0.15  # Permissive for demo


class AppConfig:
    """Application configuration loaded from environment variables."""
    
    def __init__(self):
        self._validate_required_env_vars()
        
        self.redis = RedisConfig(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            ca_cert_path=os.getenv("REDIS_CA_CERT"),
            client_cert_path=os.getenv("REDIS_CLIENT_CERT"),
            client_key_path=os.getenv("REDIS_CLIENT_KEY")
        )
        
        self.bedrock = BedrockConfig(
            region=os.getenv("AWS_REGION", "us-west-2"),
            agent_id_redis=os.getenv("BEDROCK_AGENT_ID_REDIS"),  # Redis-enhanced architecture
            agent_id_opensearch=os.getenv("BEDROCK_AGENT_ID_OPENSEARCH")  # Standard AWS architecture
        )
        
        self.cache = CacheConfig(
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
        )
    
    def _validate_required_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        required_vars = ["REDIS_HOST", "REDIS_PASSWORD", "BEDROCK_AGENT_ID_REDIS", "BEDROCK_AGENT_ID_OPENSEARCH"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please run './setup.sh' or set them manually."
            )


# Redis schema definitions - isolated from business logic
REDIS_SCHEMAS = {
    "semantic_cache": {
        "index": {
            "name": "app_semantic_cache",
            "prefix": "app_cache:",
            "storage_type": "hash"
        },
        "fields": [
            {"name": "query", "type": "text"},
            {"name": "response", "type": "text"},
            {"name": "embedding", "type": "vector", "attrs": {
                "dims": 1536,
                "distance_metric": "cosine",
                "algorithm": "flat"
            }},
            {"name": "timestamp", "type": "numeric"}
        ]
    },
    
    "conversation_memory": {
        "index": {
            "name": "app_conversation_memory",
            "prefix": "app_memory:",
            "storage_type": "hash"
        },
        "fields": [
            {"name": "session_id", "type": "text"},
            {"name": "user_message", "type": "text"},
            {"name": "bot_response", "type": "text"},
            {"name": "context", "type": "text"},
            {"name": "user_embedding", "type": "vector", "attrs": {
                "dims": 1536,
                "distance_metric": "cosine",
                "algorithm": "flat"
            }},
            {"name": "embedding", "type": "vector", "attrs": {
                "dims": 1536,
                "distance_metric": "cosine",
                "algorithm": "flat"
            }},
            {"name": "timestamp", "type": "numeric"}
        ]
    }
}

# Semantic routing query templates for financial and personal queries
ALLOWED_QUERY_REFERENCES = [
    # Personal characteristics for memory testing
    "My name is John", "My name is Sarah", "my name is john", "my name is sarah", 
    "I like pizza", "I don't like vegetables",
    "I work at Microsoft", "I am 25 years old", "What's my name?", "What do I like?",
    "Where do I live?", "What is my age?", "Tell me about myself",
    "Do you remember what I told you?", "What did I say earlier?",
    "Can you remember my personal information?", "What personal details do you know about me?",
    "Remind me what I shared with you", "My favorite color is blue", "I have two cats",
    "I drive a Tesla", "My hobby is reading",
    
    # Financial analysis queries
    "Analyze NVIDIA's financial performance", "How much money did Amazon make?",
    "what were the sales figures?", "what were yearly sales?"
    "What were Tesla's earnings last quarter?", "Show me Apple's financial results",
    "What are Amazon's quarterly earnings?", "How did Apple perform financially last year?",
    "Show me Microsoft's revenue growth", "What is Tesla's profit margin?",
    
    # Market and investment queries
    "What are the key financial risks for tech companies?", "How is the stock market performing?",
    "What's the investment outlook for cloud computing?", "Explain market volatility in financial markets",
    "What are the best dividend stocks?", "How should I invest my money?",
    "What stocks should I buy?", "Tell me about cryptocurrency investments",
    
    # Financial metrics and analysis
    "Calculate the P/E ratio for Amazon", "What is Apple's cash flow situation?",
    "Show me debt to equity ratios for tech stocks", "What's the EBITDA for Google?",
    "Analyze return on investment metrics", "What's the market cap of Microsoft?",
    "Show me financial ratios for banks",
    
    # Risk and exposure queries
    "What are Amazon's primary financial risk factors?", "Explain operational risk exposures for banks",
    "What financial challenges does Meta face?", "How do interest rates affect financial performance?",
    "What are the currency risk exposures?", "Assess investment risks for tech sector",
    
    # General financial analysis
    "Show financial analysis for tech sector", "What are the revenue projections?",
    "Explain the financial statements", "How do I analyze a company's financials?",
    "What makes a good investment?", "Explain stock valuation methods",
    "How do bond markets work?", "What drives stock prices?"
]