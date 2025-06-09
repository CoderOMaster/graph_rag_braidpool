import asyncio
import time
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# This class is provided - you don't need to implement it
class FinancialEmbeddingModel:
    """Simplified mock of a financial embedding model.
    In a production scenario, this would use a model like FinBERT.
    The current mock implementation returns constant embeddings and
    will not produce meaningful semantic similarity scores.
    """
    
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for financial text"""
        # Simulated embedding generation
        await asyncio.sleep(0.1)  # Simulates API call time
        if not text or len(text) < 10:
            return None
        # Return mock embedding (in production this would be FinBERT)
        return [0.1] * 768
    
    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Batch embedding generation"""
        return [await self.embed_text(text) for text in texts]

from transformers import AutoTokenizer, AutoModel
import torch

class FinBERTEmbeddingModel:
    """
    Embedding model using FinBERT from HuggingFace.
    Usage is similar to FinancialEmbeddingModel, but uses real FinBERT embeddings.
    """
    def __init__(self, model_name: str = "yiyanghkust/finbert-embedding"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    async def embed_text(self, text: str) -> Optional[List[float]]:
        if not text or len(text) < 10:
            return None
        # Tokenize and get embedding (mean pooling)
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            # outputs.last_hidden_state: [batch, seq_len, hidden_size]
            # Mean pooling over tokens
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return emb.tolist()

    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        # For simplicity, process sequentially (could be parallelized)
        return [await self.embed_text(text) for text in texts]

# Risk language patterns similar to our production system
RISK_PATTERNS = {
    "high_severity": [
        "material adverse effect", "significant risk", "substantial doubt", 
        "severely impact", "critical uncertainty"
    ],
    "medium_severity": [
        "could adversely affect", "may negatively impact", "potential risk",
        "uncertain outcome", "possible limitation"
    ],
    "low_severity": [
        "may affect", "could influence", "potential consideration",
        "possible variation", "slight uncertainty"
    ]
}

# Weights for each severity category
SEVERITY_WEIGHTS = {
    "high_severity": 3.0,
    "medium_severity": 2.0,
    "low_severity": 1.0
}

# Common hedging words to look for (they tend to soften language)
HEDGING_WORDS = {"may", "could", "potentially", "possible", "might", "likely"}

async def calculate_risk_language_drift(
    current_text: str, 
    previous_text: str,
    embedding_model: FinancialEmbeddingModel
) -> Dict[str, Any]:
    """
    Calculate the drift in risk language between current and previous text.
    
    Args:
        current_text: Text from current filing
        previous_text: Text from previous filing
        embedding_model: Model for generating embeddings
        
    Returns:
        Dictionary containing:
        - risk_drift_score: Float between 0-1 (higher means more intensified risk)
        - confidence: Confidence in the calculation (0-1)
        - evidence: List of specific phrases showing risk language changes
        - processing_time_ms: Processing time in milliseconds
    """
    start_time = time.perf_counter()
    evidence: List[str] = []
    
    # --- STEP 1: Count and weight occurrences of each risk pattern in both texts, considering hedging ---
    # HEDGING_DISCOUNT_FACTOR: e.g., 0.6 means a 40% reduction in weight if hedged. Applied if a hedging word is found within HEDGING_WINDOW_CHARS before a risk phrase.
    HEDGING_DISCOUNT_FACTOR = 0.6 
    # HEDGING_WINDOW_CHARS: How many characters before a risk phrase to check for hedging words.
    HEDGING_WINDOW_CHARS = 30

    def count_weighted_risks(text: str) -> Tuple[float, Dict[str, int]]:
        """
        Counts raw occurrences of risk phrases and calculates an adjusted weighted score.
        The weighted score for each instance of a risk phrase is reduced if hedging 
        language (from HEDGING_WORDS) is detected in the HEDGING_WINDOW_CHARS 
        immediately preceding it.

        Args:
            text (str): The input text to analyze.

        Returns:
            Tuple[float, Dict[str, int]]: 
                - total_adjusted_weighted_score: The sum of weights for all risk phrases, 
                  adjusted for hedging.
                - phrase_raw_counts: A dictionary mapping each detected risk phrase to its 
                  raw (unweighted, unadjusted) count in the text.
        """
        total_adjusted_weighted_score = 0.0
        phrase_raw_counts: Dict[str, int] = {}
        text_lower = text.lower()
        
        for severity, phrases_list in RISK_PATTERNS.items():
            base_weight = SEVERITY_WEIGHTS.get(severity, 1.0)
            for phrase in phrases_list:
                # Use word-boundary matching for exact phrase
                pattern = rf"\b{re.escape(phrase)}\b"
                
                # Find all match objects to get their positions for hedging check
                # and to count raw occurrences for evidence
                current_phrase_matches = list(re.finditer(pattern, text_lower, flags=re.IGNORECASE))
                num_raw_matches_for_phrase = len(current_phrase_matches)
                
                if num_raw_matches_for_phrase > 0:
                    # Store raw counts for evidence (accumulate if phrase appears in multiple severity lists, though unlikely with current RISK_PATTERNS)
                    phrase_raw_counts[phrase] = phrase_raw_counts.get(phrase, 0) + num_raw_matches_for_phrase
                    
                    # Calculate adjusted weight for each individual match of the phrase
                    for match_obj in current_phrase_matches:
                        start_index = match_obj.start() # Start index of the current risk phrase match
                        
                        # Define window to check for hedging words (immediately preceding the phrase)
                        window_start = max(0, start_index - HEDGING_WINDOW_CHARS)
                        preceding_text_window = text_lower[window_start:start_index]
                        
                        is_hedged = False
                        for hedge_word in HEDGING_WORDS:
                            # Check for whole word match of hedge_word in the window
                            if re.search(rf"\b{re.escape(hedge_word)}\b", preceding_text_window):
                                is_hedged = True
                                break
                        
                        current_instance_weight = base_weight
                        if is_hedged:
                            current_instance_weight *= HEDGING_DISCOUNT_FACTOR
                            
                        total_adjusted_weighted_score += current_instance_weight
                        
        return total_adjusted_weighted_score, phrase_raw_counts

    # Get weighted scores (adjusted for hedging) and raw counts for both texts
    weighted_current, raw_counts_current = count_weighted_risks(current_text)
    weighted_previous, raw_counts_previous = count_weighted_risks(previous_text)

    # --- STEP 2: Build evidence list based on increases in specific phrases (using raw counts) ---
    for phrase, curr_raw_count in raw_counts_current.items():
        prev_raw_count = raw_counts_previous.get(phrase, 0)
        diff = curr_raw_count - prev_raw_count
        if diff > 0:
            # Evidence now refers to raw count changes, not potentially weighted/adjusted counts
            evidence.append(f'"{phrase}" raw count increased by {diff}')
    
    # If no exact phrase increase but overall weight increased, show summary
    if not evidence and weighted_current > weighted_previous:
        evidence.append(f"Total weighted risk count increased from {weighted_previous:.1f} to {weighted_current:.1f}")

    # --- STEP 3: Generate embeddings and compute semantic similarity ---
    try:
        embed_curr = await embedding_model.embed_text(current_text)
        print(f"Embedding for current text: {embed_curr}")
    except Exception:
        embed_curr = None
    try:
        embed_prev = await embedding_model.embed_text(previous_text)
        print(f"Embedding for previous text: {embed_prev}")
    except Exception:
        embed_prev = None

    if embed_curr is None or embed_prev is None:
        # If embedding generation fails, set a neutral semantic score
        cosine_sim = 0.5
    else:
        vec_curr = np.array(embed_curr, dtype=np.float32)
        vec_prev = np.array(embed_prev, dtype=np.float32)
        norm_curr = np.linalg.norm(vec_curr)
        norm_prev = np.linalg.norm(vec_prev)
        if norm_curr == 0 or norm_prev == 0:
            cosine_sim = 0.5
        else:
            cosine_sim = float(np.dot(vec_curr, vec_prev) / (norm_curr * norm_prev))

    # A higher 1 - cosine_sim implies more semantic drift
    semantic_score = max(0.0, min(1.0, 1.0 - cosine_sim))

    # --- STEP 4: Compute a count-based drift score ---
    if weighted_current <= weighted_previous:
        count_score = 0.0
    else:
        if weighted_previous <= 0:
            count_score = 1.0
        else:
            raw_diff = weighted_current - weighted_previous
            count_score = min(raw_diff / weighted_previous, 1.0)

    # --- STEP 5: Combine both signals into a final risk drift score ---
    risk_drift_score = float((semantic_score + count_score) / 2.0)

    # --- STEP 6: Estimate confidence in the result ---
    # Embedding confidence: 1.0 if both embeddings exist, else 0.0
    embed_conf = 1.0 if (embed_curr is not None and embed_prev is not None) else 0.0
    # Evidence confidence: up to 1.0 based on number of evidence items (cap at 5+ items)
    evidence_conf = min(len(evidence) / 5.0, 1.0)
    # Weighted combination
    confidence = float(min(1.0, embed_conf * 0.7 + evidence_conf * 0.3))

    # --- STEP 7: Measure processing time ---
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "risk_drift_score": round(risk_drift_score, 4),
        "confidence": round(confidence, 4),
        "evidence": evidence,
        "processing_time_ms": round(elapsed_ms, 2)
    }

# Example Usage (at the end of your file, or in another script)
async def main():
    print("--- Using Mock Model (fast, constant embeddings) ---")
    model = FinancialEmbeddingModel()
    text1 = "This is a test text with some potential risk."
    text2 = "This is another test, and it may affect our results significantly, creating a substantial doubt."
    results = await calculate_risk_language_drift(text2, text1, model)
    import json
    print(json.dumps(results, indent=4))
    
    print("\n--- Using Real FinBERT Model (from HuggingFace) ---")
    finbert_model = FinBERTEmbeddingModel()
    results_finbert = await calculate_risk_language_drift(text2, text1, finbert_model)
    print(json.dumps(results_finbert, indent=4))

if __name__ == "__main__":
    asyncio.run(main())