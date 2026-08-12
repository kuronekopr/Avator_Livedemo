import json
import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from server.config import settings

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAGEngine")


class RuriEmbeddingFunction:
    """
    日本語 Embedding モデル ruri-v3 (cl-nagoya/ruri-v3-310m) 用 ChromaDB カスタム Embedding Function
    """
    def __init__(self, model_name: str = "cl-nagoya/ruri-v3-310m"):
        logger.info(f"Loading Japanese Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding Model loaded successfully.")

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """クエリ埋め込み (query: プレフィックス付与)"""
        formatted_queries = [
            f"query: {text}" if not text.startswith("query:") else text
            for text in input
        ]
        embeddings = self.model.encode(
            formatted_queries,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """ドキュメント埋め込み (passage: プレフィックス付与)"""
        formatted_docs = [
            f"passage: {text}" if not text.startswith("passage:") else text
            for text in input
        ]
        embeddings = self.model.encode(
            formatted_docs,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)


class RAGEngine:
    def __init__(self, qa_json_path: str):
        self.qa_json_path = qa_json_path
        self.documents: List[Dict[str, Any]] = []
        self.is_chroma_active: bool = False
        self.chroma_collection = None
        self.last_modified_time: float = 0.0
        
        self.ruri_ef: Optional[RuriEmbeddingFunction] = None
        self.chroma_client: Optional[chromadb.Client] = None
        
        # Gemini API の初期化
        self.gemini_model = None
        self.init_gemini_llm()

        self.reload_and_reindex()

    def init_gemini_llm(self):
        """.env や環境変数に GEMINI_API_KEY が定義されている場合、Gemini API を初期化"""
        api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key and api_key.strip():
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key.strip())
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini API LLM successfully configured from .env file.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API LLM: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
            logger.info("No GEMINI_API_KEY found in .env. Intelligent RAG context formatter will be used as fallback.")

    def check_and_auto_reload(self):
        """ファイルの更新日時をチェックし、変更されていれば自動再ロード"""
        if os.path.exists(self.qa_json_path):
            current_mtime = os.path.getmtime(self.qa_json_path)
            if self.last_modified_time != 0.0 and current_mtime > self.last_modified_time:
                logger.info(f"Detected update in {self.qa_json_path}. Auto-reindexing Vector DB...")
                self.reload_and_reindex()

    def reload_and_reindex(self) -> Dict[str, Any]:
        """
        globallogic_qa.json を再読み込みし、Vector DB (ruri-v3 ChromaDB) を初期化・再インデックスする
        """
        if not os.path.exists(self.qa_json_path):
            raise FileNotFoundError(f"QA data file not found at: {self.qa_json_path}")
        
        # .env の再読み込みチェック
        self.init_gemini_llm()

        with open(self.qa_json_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        self.last_modified_time = os.path.getmtime(self.qa_json_path)
        logger.info(f"Loaded {len(self.documents)} QA items from {self.qa_json_path}")

        if self.ruri_ef is None:
            self.ruri_ef = RuriEmbeddingFunction("cl-nagoya/ruri-v3-310m")

        if self.chroma_client is None:
            self.chroma_client = chromadb.Client()

        collection_name = settings.COLLECTION_NAME
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Resetting existing Vector DB collection '{collection_name}'...")
        except Exception:
            pass

        self.chroma_collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=self.ruri_ef,
            metadata={"hnsw:space": "cosine"}
        )

        ids = [str(item["id"]) for item in self.documents]
        documents_text = [
            f"カテゴリ: {item['Category']}\n質問: {item['Question']}\n回答: {item['Answer']}"
            for item in self.documents
        ]
        metadatas = [
            {
                "category": item["Category"],
                "question": item["Question"],
                "answer": item["Answer"],
                "qa_id": item["id"]
            }
            for item in self.documents
        ]

        logger.info(f"Re-indexing {len(documents_text)} documents into Vector DB...")
        self.chroma_collection.add(
            ids=ids,
            documents=documents_text,
            metadatas=metadatas
        )

        self.is_chroma_active = True
        logger.info(f"Successfully re-indexed Vector DB with {len(self.documents)} QA items.")
        
        return {
            "status": "success",
            "qa_count": len(self.documents),
            "collection_name": collection_name,
            "message": f"Successfully re-indexed Vector DB with {len(self.documents)} QA items."
        }

    def search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        ruri-v3 ベクトルデータベースによるコサイン類似度検索
        """
        self.check_and_auto_reload()

        if not query or not query.strip():
            return []

        results = self.chroma_collection.query(
            query_texts=[query.strip()],
            n_results=top_k
        )

        formatted_results = []
        if results and "metadatas" in results and len(results["metadatas"]) > 0:
            for i, meta in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                score = round(max(0.0, 1.0 - float(distance)), 4)
                formatted_results.append({
                    "id": meta["qa_id"],
                    "category": meta["category"],
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "score": score
                })

        return formatted_results

    def generate_rag_response(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        RAG (Retrieval-Augmented Generation) 検索と LLM による自然な対話応答文の生成
        """
        clean_q = query.strip()
        if not clean_q:
            return {
                "generated_text": "質問を入力してください。",
                "search_results": []
            }

        greeting_keywords = ["あなたは何", "何ができる", "自己紹介", "だれ", "誰", "説明して", "何について", "教えてくれるの"]
        is_greeting_or_intro = any(k in clean_q for k in greeting_keywords) and len(clean_q) < 20

        search_results = self.search_knowledge_base(query=clean_q, top_k=top_k)

        # .env に Gemini API Key が設定されている場合
        if self.gemini_model:
            try:
                context_str = "\n\n".join([
                    f"[参照ナレッジ {i+1}] (カテゴリ: {r['category']})\nQ: {r['question']}\nA: {r['answer']}"
                    for i, r in enumerate(search_results)
                ])

                system_prompt = f"""あなたは GlobalLogic Japan の公式AIアバターアシスタントです。
ユーザーからの質問に対して、以下の【参照ナレッジ】の情報を基に、親しみやすく丁寧で自然な会話文で答えてください。

【注意事項】
- ユーザーの質問に直接答える会話文を作成してください。単にナレッジの文章をコピペせず、文脈に合った自然な応答に組み立ててください。
- ユーザーが「あなたは何を説明できるの？」「何ができるの？」と聞いた場合は、GlobalLogic Japanの会社概要、サービス（デジタルエンジニアリング、AI/VelocityAI、IT/OTトランスフォーメーション等）をご案内できる旨を親切に回答してください。
- 事例や具体的な質問には、参照ナレッジ内の適切な回答を要約・引用して分かりやすく説明してください。

【参照ナレッジ】
{context_str}

【ユーザーの質問】
{clean_q}

【回答文】"""

                response = self.gemini_model.generate_content(system_prompt)
                if response and response.text:
                    return {
                        "generated_text": response.text.strip(),
                        "search_results": search_results
                    }
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")

        # フォールバック (APIキー未設定時)
        if is_greeting_or_intro:
            generated_text = (
                "私は GlobalLogic Japan の AI 公式アシスタントです！"
                "弊社の会社概要をはじめ、デジタルエンジニアリング、AI（VelocityAI）、ソフトウェア開発、"
                "IT/OTトランスフォーメーション、ならびに各種導入事例や強みについて分かりやすくお答えいたします。"
                "どのようなことでもお気軽にご質問ください。"
            )
        elif search_results:
            top = search_results[0]
            if "事例" in clean_q or "実績" in clean_q:
                generated_text = f"【事例・実績のご紹介】\n{top['answer']}"
            elif "強み" in clean_q or "特徴" in clean_q:
                generated_text = f"【GlobalLogicの強み・特徴】\n{top['answer']}"
            else:
                generated_text = f"ご質問の「{top['question']}」についてお答えいたします。\n\n{top['answer']}"
        else:
            generated_text = "申し訳ありません。ご質問に関する該当情報が見つかりませんでした。キーワードを変えて再度お尋ねください。"

        return {
            "generated_text": generated_text,
            "search_results": search_results
        }
