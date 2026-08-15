"""
MSMARCO-XI Local Dataset Generator & Explorer
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

This script populates data/msmarco_sample.json with structured MSMARCO-XI records 
matching the official dataset schema fields for reproducible development and testing.
"""

import os
import json
import statistics

SAMPLE_DATASET = [
    {
        "query_id": "101",
        "query": "what is retrieval augmented generation in artificial intelligence",
        "passage_id": "p101",
        "passage": "Retrieval-Augmented Generation (RAG) is an architectural pattern in AI that enhances Large Language Models by retrieving relevant factual documents from an external knowledge base before generating an answer. RAG reduces hallucinations and grounds responses in verified data.",
        "url": "https://ai.example.org/rag-explanation",
        "language": "en"
    },
    {
        "query_id": "102",
        "query": "how vector embeddings work in semantic search",
        "passage_id": "p102",
        "passage": "Vector embeddings represent text as dense mathematical vectors in a high-dimensional space. Words or sentences with similar semantic meanings are placed close together in this vector space. Semantic search uses distance metrics like cosine similarity or dot product to find relevant documents even when exact keyword matches are absent.",
        "url": "https://ai.example.org/embeddings",
        "language": "en"
    },
    {
        "query_id": "103",
        "query": "what is speech to text latency optimization",
        "passage_id": "p103",
        "passage": "Speech-to-Text (STT) latency optimization involves streaming audio buffers over WebSocket connections, selecting lightweight acoustic models like Sarvam AI or ElevenLabs, and utilizing VAD (Voice Activity Detection) to process speech chunks immediately rather than waiting for silence.",
        "url": "https://ai.example.org/stt-latency",
        "language": "en"
    },
    {
        "query_id": "104",
        "query": "what is FAISS vector search library",
        "passage_id": "p104",
        "passage": "FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that may not fit in RAM. It offers GPU support and HNSW indexing for sub-millisecond search.",
        "url": "https://ai.example.org/faiss",
        "language": "en"
    },
    {
        "query_id": "105",
        "query": "where is Goa located in India",
        "passage_id": "p105",
        "passage": "Goa is a state located on the southwestern coast of India within the Konkan region. It is bounded by Maharashtra to the north and Karnataka to the east and south, with the Arabian Sea forming its western coast. It is India's smallest state by area.",
        "url": "https://geography.example.org/goa",
        "language": "en"
    },
    {
        "query_id": "106",
        "query": "what is Sarvam AI known for",
        "passage_id": "p106",
        "passage": "Sarvam AI is an Indian AI research organization focused on building generative AI models and voice infrastructure tailored for Indic languages and regional accents. Sarvam provides fast Speech-to-Text (STT) APIs optimized for Indian English and native Indian languages.",
        "url": "https://ai.example.org/sarvam",
        "language": "en"
    },
    {
        "query_id": "107",
        "query": "how to measure P50 P70 and P100 latency in software systems",
        "passage_id": "p107",
        "passage": "P50 latency represents the median response time where 50% of requests are faster. P70 is the 70th percentile, and P100 represents the worst-case maximum latency recorded across all requests. Percentile measurement helps identify long-tail network and model delays.",
        "url": "https://systems.example.org/latency-metrics",
        "language": "en"
    },
    {
        "query_id": "108",
        "query": "what are RAG guardrails and hallucination prevention",
        "passage_id": "p108",
        "passage": "Guardrails in RAG pipelines validate incoming user prompts and outgoing generated answers. Grounding guardrails verify that every statement in the generated answer is directly supported by retrieved context passages, rejecting ungrounded or off-topic responses.",
        "url": "https://ai.example.org/guardrails",
        "language": "en"
    },
    {
        "query_id": "109",
        "query": "multilingual e5 small embedding model features",
        "passage_id": "p109",
        "passage": "intfloat/multilingual-e5-small is a 12-layer Transformer embedding model trained on multilingual text pairs. It maps queries and passages into a shared 384-dimensional vector space with prefix instructions like 'query: ' and 'passage: ' for asymmetric retrieval.",
        "url": "https://huggingface.co/intfloat/multilingual-e5-small",
        "language": "en"
    },
    {
        "query_id": "110",
        "query": "what is Google Gemini 1.5 flash model",
        "passage_id": "p110",
        "passage": "Google Gemini 1.5 Flash is a lightweight, multimodal model optimized for high frequency, low latency tasks. It features a massive context window of up to 1 million tokens and strong structured JSON extraction abilities.",
        "url": "https://ai.google.dev/gemini",
        "language": "en"
    }
]

def prepare_dataset():
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "msmarco_sample.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_DATASET, f, indent=2, ensure_ascii=False)
        
    print("=" * 60)
    print("MSMARCO-XI Sample Dataset Analysis")
    print("=" * 60)
    print(f"[+] Written dataset file: {file_path}")
    print(f"[+] Total Records: {len(SAMPLE_DATASET)}")
    print(f"[*] Available Fields: {list(SAMPLE_DATASET[0].keys())}")
    
    char_lens = [len(item["passage"]) for item in SAMPLE_DATASET]
    word_lens = [len(item["passage"].split()) for item in SAMPLE_DATASET]
    
    print("\nPassage Quantitative Statistics:")
    print(f"  - Min Passage Length : {min(char_lens)} chars ({min(word_lens)} words)")
    print(f"  - Max Passage Length : {max(char_lens)} chars ({max(word_lens)} words)")
    print(f"  - Avg Passage Length : {statistics.mean(char_lens):.1f} chars ({statistics.mean(word_lens):.1f} words)")
    print("=" * 60)

if __name__ == "__main__":
    prepare_dataset()
