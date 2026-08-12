import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from server.rag_engine import RAGEngine
from server.config import settings

def main():
    print("=== Re-indexing Vector DB from globallogic_qa.json ===")
    rag = RAGEngine(qa_json_path=settings.QA_JSON_PATH)
    result = rag.reload_and_reindex()
    print(f"\n[Success] {result['message']}")
    print(f"Total QA Items indexed: {result['qa_count']}")
    print("Vector DB Collection:", result['collection_name'])

if __name__ == "__main__":
    main()
