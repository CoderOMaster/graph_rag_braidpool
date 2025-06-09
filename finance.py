import asyncio
import time
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import json

class FinancialEmbeddingModel:
    """Simplified version of our production embedding model"""
    
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
    
    # --- STEP 1: Count and weight occurrences of each risk pattern in both texts ---
    def count_weighted_risks(text: str) -> Tuple[float, Dict[str, int]]:
        total_weighted = 0.0
        counts: Dict[str, int] = {}
        text_lower = text.lower()
        
        for severity, phrases in RISK_PATTERNS.items():
            weight = SEVERITY_WEIGHTS.get(severity, 1.0)
            for phrase in phrases:
                # Use word-boundary matching for exact phrase
                pattern = rf"\b{re.escape(phrase)}\b"
                matches = re.findall(pattern, text_lower, flags=re.IGNORECASE)
                num_matches = len(matches)
                if num_matches > 0:
                    counts[phrase] = num_matches
                    total_weighted += weight * num_matches
        return total_weighted, counts

    weighted_current, counts_current = count_weighted_risks(current_text)
    weighted_previous, counts_previous = count_weighted_risks(previous_text)

    # --- STEP 2: Build evidence list based on increases in specific phrases ---
    for phrase, curr_count in counts_current.items():
        prev_count = counts_previous.get(phrase, 0)
        diff = curr_count - prev_count
        if diff > 0:
            evidence.append(f'"{phrase}" count increased by {diff}')
    
    # If no exact phrase increase but overall weight increased, show summary
    if not evidence and weighted_current > weighted_previous:
        evidence.append(f"Total weighted risk count increased from {weighted_previous:.1f} to {weighted_current:.1f}")

    # --- STEP 3: Generate embeddings and compute semantic similarity ---
    try:
        embed_curr = await embedding_model.embed_text(current_text)
    except Exception:
        embed_curr = None
    try:
        embed_prev = await embedding_model.embed_text(previous_text)
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

async def main():
    model = FinancialEmbeddingModel()
    text1 = "This is a test text with some potential risk."
    text2 = "This is another test, and it may affect our results significantly, creating a substantial doubt."
    results = await calculate_risk_language_drift(text2, text1, model)
    print(json.dumps(results, indent=4))
    
if __name__ == "__main__":
    asyncio.run(main())