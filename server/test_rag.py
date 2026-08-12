import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from server.rag_engine import RAGEngine
from server.config import settings

def main():
    print("=== Testing High-Precision Hybrid RAG Engine ===")
    rag = RAGEngine(qa_json_path=settings.QA_JSON_PATH)

    test_queries = [
        "日本の事業所はどこにありますか",
        "事業所はどこにありますか",
        "GlobalLogicの親会社はどこですか？",
        "VelocityAIとは何ですか？どのようなメリットがありますか？",
        "開発に対応しているプログラミング言語は？",
        "モダナイゼーションにかかる期間の目安は？"
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n--- [Query {i}] \"{q}\" ---")
        results = rag.search_knowledge_base(query=q, top_k=2)
        for r in results:
            print(f"  > Score: {r['score']:.4f} | [ID:{r['id']}] Category: {r['category']}")
            print(f"    Q: {r['question']}")
            print(f"    A: {r['answer'][:80]}...\n")

if __name__ == "__main__":
    main()
