"""
Embedding utilities - Generate vector embeddings.

Two backends are supported:
1. SentenceTransformers (local) - loads Qwen3 / standard models locally. This is
   the default path when no ``EMBEDDING_BASE_URL`` is set.
2. OpenAI-compatible Embeddings API - when ``EMBEDDING_BASE_URL`` is configured
   (see config.py), embeddings are requested from a deployed service (e.g. a vLLM
   server). This keeps the embedding weights served exactly once and reuses the
   already-deployed Qwen3-Embedding model instead of loading it a second time.

The API backend L2-normalizes its outputs to match SentenceTransformers'
``normalize_embeddings=True`` behaviour used by the local path.
"""
from typing import List, Optional, Dict, Any
import numpy as np
from simplemem.core.settings import settings as config
import os


class EmbeddingModel:
    """
    Embedding model. Uses a deployed OpenAI-compatible embeddings service when
    ``EMBEDDING_BASE_URL`` is configured, otherwise loads locally via
    SentenceTransformers.
    """
    def __init__(self, model_name: str = None, use_optimization: bool = True):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization

        # --- Deployed embeddings API mode ---------------------------------
        api_base = getattr(config, "EMBEDDING_BASE_URL", None)
        if api_base:
            self._init_api(api_base)
            return

        print(f"Loading embedding model: {self.model_name}")

        # Check if it's a Qwen3 model (through SentenceTransformers)
        if self.model_name.startswith("qwen3"):
            self._init_qwen3_sentence_transformer()
        else:
            self._init_standard_sentence_transformer()

    # ------------------------------------------------------------------
    # API backend
    # ------------------------------------------------------------------
    def _init_api(self, api_base: str):
        from openai import OpenAI
        api_key = getattr(config, "EMBEDDING_API_KEY", None) or "EMPTY"
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        # Probe a single input to discover the embedding dimension.
        probe = self.client.embeddings.create(model=self.model_name, input=["probe"])
        self.dimension = len(probe.data[0].embedding)
        self.model_type = "api_embedding"
        self.supports_query_prompt = False
        self.query_instruction = getattr(config, "EMBEDDING_QUERY_INSTRUCTION", "") or ""
        print(f"Embedding API ready: base={api_base} model={self.model_name} dim={self.dimension}")

    def _api_encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        batch_size = 256
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=chunk)
            vecs.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
        arr = np.stack(vecs, axis=0)
        # L2-normalize (mirrors SentenceTransformers normalize_embeddings=True)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr

    # ------------------------------------------------------------------
    # SentenceTransformer backends (local)
    # ------------------------------------------------------------------
    def _init_qwen3_sentence_transformer(self):
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer

            # Map model names to actual model paths
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }

            model_path = qwen3_models.get(self.model_name, self.model_name)
            print(f"Loading Qwen3 model via SentenceTransformers: {model_path}")

            # Initialize with optimization settings
            if self.use_optimization:
                try:
                    # Try to use flash_attention_2 and left padding for better performance
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2",
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                    print("Qwen3 loaded with flash_attention_2 optimization")
                except Exception as e:
                    print(f"Flash attention failed ({e}), using standard loading...")
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)

            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"

            # Check if Qwen3 supports query prompts
            self.supports_query_prompt = hasattr(self.model, 'prompts') and 'query' in getattr(self.model, 'prompts', {})

            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")

        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            print("Falling back to default SentenceTransformers model...")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self):
        """Initialize standard SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _fallback_to_sentence_transformer(self):
        """Fallback to default SentenceTransformer model"""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    # ------------------------------------------------------------------
    # Public encode interface (dispatches to the active backend)
    # ------------------------------------------------------------------
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        if getattr(self, "model_type", "") == "api_embedding":
            return self._api_encode(texts, is_query=is_query)

        # Use query prompt for Qwen3 models when encoding queries
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        else:
            return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """Encode single text."""
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: List[str]) -> np.ndarray:
        """Encode queries with optimal settings for Qwen3."""
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents (no query prompt)."""
        return self.encode(documents, is_query=False)

    # ------------------------------------------------------------------
    # SentenceTransformer encoders
    # ------------------------------------------------------------------
    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3 query prompt"""
        try:
            embeddings = self.model.encode(
                texts,
                prompt_name="query",  # Use Qwen3's query prompt
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Query prompt encoding failed: {e}, falling back to standard encoding")
            return self._encode_standard(texts)

    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using standard method"""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings
