"""
MSMARCO-XI Dataset Exploration Script
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

This script inspects the schema, sample records, and passage length statistics
of the AI4Bharat MSMARCO-XI dataset from Hugging Face.
"""

import os
import json
import statistics
from typing import Dict, Any, List

def explore_hf_dataset():
    print("=" * 60)
    print("AI4Bharat MSMARCO-XI Dataset Exploration")
    print("=" * 60)
    
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        print("[!] The 'datasets' library is required. Install via: pip install datasets")
        return

    dataset_name = "ai4bharat/MSMARCO-XI"
    print(f"[*] Fetching dataset configs for '{dataset_name}'...")
    
    try:
        configs = get_dataset_config_names(dataset_name)
        print(f"[+] Available language/config splits: {configs}")
    except Exception as e:
        print(f"[!] Could not list configs directly: {e}")
        configs = ["default"]

    selected_config = configs[0] if configs else "english"
    print(f"[*] Loading streaming sample from config: '{selected_config}'...")
    
    try:
        # Load in streaming mode to inspect without downloading full gigabytes
        ds = load_dataset(dataset_name, selected_config, split="train", streaming=True)
        
        sample_records: List[Dict[str, Any]] = []
        passage_lengths_char: List[int] = []
        passage_lengths_word: List[int] = []
        
        count = 0
        max_samples = 100
        
        for item in ds:
            sample_records.append(item)
            
            # Find passage text field across possible schema keys
            passage_text = ""
            for key in ["passage", "passage_text", "text", "segment", "query"]:
                if key in item and isinstance(item[key], str) and item[key].strip():
                    passage_text = item[key]
                    break
            
            if passage_text:
                passage_lengths_char.append(len(passage_text))
                passage_lengths_word.append(len(passage_text.split()))
            
            count += 1
            if count >= max_samples:
                break
                
        print("\n" + "-" * 60)
        print(f"[+] Sample Inspection Summary ({len(sample_records)} records loaded)")
        print("-" * 60)
        
        if sample_records:
            first_item = sample_records[0]
            print(f"[*] Schema Fields Available: {list(first_item.keys())}")
            print("\n[*] Sample Record 1 Preview:")
            for k, v in first_item.items():
                val_str = str(v)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "... [truncated]"
                print(f"    - {k}: {val_str}")
        
        if passage_lengths_char:
            avg_char = statistics.mean(passage_lengths_char)
            max_char = max(passage_lengths_char)
            min_char = min(passage_lengths_char)
            avg_words = statistics.mean(passage_lengths_word)
            
            print("\n" + "-" * 60)
            print("[+] Passage Length Statistics:")
            print("-" * 60)
            print(f"    - Min Character Length : {min_char}")
            print(f"    - Max Character Length : {max_char}")
            print(f"    - Avg Character Length : {avg_char:.2f}")
            print(f"    - Avg Word Count       : {avg_words:.2f}")
        
        # Save sample subset to data/msmarco_sample.json for fast offline testing
        output_dir = os.path.join(os.path.dirname(__file__), "../data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "msmarco_sample.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_records, f, indent=2, ensure_ascii=False)
            
        print(f"\n[+] Saved sample dataset subset ({len(sample_records)} items) to: {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"[!] Error inspecting dataset: {e}")
        # Fallback sample dataset generation for offline robust setup
        create_offline_fallback_sample()

def create_offline_fallback_sample():
    print("[*] Creating offline MSMARCO-XI sample records for local development...")
    output_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "msmarco_sample.json")
    
    fallback_records = [
        {
            "query_id": "1",
            "query": "what is retrieval augmented generation",
            "passage_id": "p1",
            "passage": "Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM responses by grounding the model on external sources of knowledge. RAG combines information retrieval with text generation.",
            "url": "https://example.org/rag-overview",
            "language": "en"
        },
        {
            "query_id": "2",
            "query": "how vector database works",
            "passage_id": "p2",
            "passage": "A vector database indexes and stores high-dimensional vector embeddings for fast similarity search. It enables applications to perform Approximate Nearest Neighbor (ANN) search across millions of documents in milliseconds.",
            "url": "https://example.org/vector-db",
            "language": "en"
        },
        {
            "query_id": "3",
            "query": "speech to text latency optimization",
            "passage_id": "p3",
            "passage": "Optimizing speech-to-text latency involves streaming audio chunks directly over WebSockets or HTTP streams to STT engines like Sarvam AI or ElevenLabs. Sub-100ms response rates require low sample rate audio buffers.",
            "url": "https://example.org/stt-latency",
            "language": "en"
        }
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fallback_records, f, indent=2, ensure_ascii=False)
    print(f"[+] Fallback sample saved to {output_path}")

if __name__ == "__main__":
    explore_hf_dataset()
