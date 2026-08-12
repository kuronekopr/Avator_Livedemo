import json
import os
import time
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
        
        # セッション会話履歴の保持構造
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.SESSION_TIMEOUT_SECONDS = 180  # 3分無操作で自動初期化

        # Gemini API LLM の初期化
        self.gemini_model = None
        self.init_gemini_llm()

        self.reload_and_reindex()

    def init_gemini_llm(self):
        """Gemini API LLM の初期化 (gemini-2.0-flash / gemini-1.5-flash サポート)"""
        api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key and api_key.strip():
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key.strip())
                # 利用可能な最新の Flash モデルを選択
                for model_candidate in ["gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_candidate)
                        logger.info(f"Gemini API LLM ({model_candidate}) successfully configured from .env file.")
                        break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API LLM: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
            logger.info("No GEMINI_API_KEY found in .env. Intelligent RAG context formatter will be used as fallback.")

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        """指定されたセッションIDの会話履歴を初期化"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        logger.info(f"[SessionManager] Session '{session_id}' has been reset.")
        return {
            "status": "success",
            "session_id": session_id,
            "message": "ご利用ありがとうございました。セッションを終了・初期化しました。"
        }

    def _get_or_clean_session(self, session_id: str) -> List[Dict[str, str]]:
        """セッションの取得および 3 分無操作タイムアウト判定・初期化"""
        now = time.time()
        if session_id in self.sessions:
            last_active = self.sessions[session_id].get("last_active", now)
            if now - last_active > self.SESSION_TIMEOUT_SECONDS:
                logger.info(f"[SessionManager] Session '{session_id}' timed out after 3 minutes of inactivity. Resetting...")
                self.reset_session(session_id)
                self.sessions[session_id] = {"history": [], "last_active": now}
            else:
                self.sessions[session_id]["last_active"] = now
        else:
            self.sessions[session_id] = {"history": [], "last_active": now}

        return self.sessions[session_id]["history"]

    def check_and_auto_reload(self):
        """ファイルの更新日時をチェックし、変更されていれば自動再ロード"""
        if os.path.exists(self.qa_json_path):
            current_mtime = os.path.getmtime(self.qa_json_path)
            if self.last_modified_time != 0.0 and current_mtime > self.last_modified_time:
                logger.info(f"Detected update in {self.qa_json_path}. Auto-reindexing Vector DB...")
                self.reload_and_reindex()

    def reload_and_reindex(self) -> Dict[str, Any]:
        """globallogic_qa.json を再読み込みし Vector DB を初期化"""
        if not os.path.exists(self.qa_json_path):
            raise FileNotFoundError(f"QA data file not found at: {self.qa_json_path}")
        
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
        """ruri-v3 ベクトルデータベースによるコサイン類似度検索"""
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

    def generate_rag_response(self, query: str, session_id: str = "default_session", top_k: int = 3) -> Dict[str, Any]:
        """
        マルチターン会話履歴を引き継いだ RAG (検索＋生成) 対話応答処理
        """
        clean_q = query.strip()
        if not clean_q:
            return {
                "generated_text": "質問を入力してください。",
                "search_results": []
            }

        # 1. 会話履歴の取得と 3 分無操作チェック
        history = self._get_or_clean_session(session_id)

        # 文脈補正クエリ
        expanded_query = clean_q
        if history and len(clean_q) <= 10:
            last_user_q = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
            if last_user_q:
                expanded_query = f"{last_user_q} {clean_q}"

        # 2. Vector DB からの知識検索
        search_results = self.search_knowledge_base(query=expanded_query, top_k=top_k)

        # 3. 会話履歴テキストのフォーマット
        recent_history = history[-8:] if len(history) > 8 else history
        history_str = "\n".join([
            f"{'ユーザー' if h['role']=='user' else 'AIアシスタント'}: {h['content']}"
            for h in recent_history
        ]) if recent_history else "（これまでの会話はありません）"

        # 4. LLM 生成処理 (Gemini API が有効な場合)
        if self.gemini_model:
            try:
                context_str = "\n\n".join([
                    f"[参照ナレッジ {i+1}] (カテゴリ: {r['category']})\nQ: {r['question']}\nA: {r['answer']}"
                    for i, r in enumerate(search_results)
                ])

                system_prompt = f"""あなたは GlobalLogic Japan の公式AIアバターアシスタントです。
これまでの【過去の会話履歴】と【参照ナレッジ】を踏まえ、ユーザーの最新の質問に親しみやすく丁寧で自然に答えてください。

【注意事項】
- 直前の会話履歴を踏まえ、「その会社」「それ」などの指示語がある場合は直前の文脈を理解して的確に回答してください。
- 参照ナレッジの情報を優先して正確に説明してください。
- 「あなたは何を説明できるの？」といった挨拶・自己紹介には、GlobalLogic Japanの概要やサービスを案内してください。

【過去の会話履歴】
{history_str}

【参照ナレッジ】
{context_str}

【ユーザーの最新の質問】
{clean_q}

【回答文】"""

                response = self.gemini_model.generate_content(system_prompt)
                if response and response.text:
                    generated_text = response.text.strip()
                    history.append({"role": "user", "content": clean_q})
                    history.append({"role": "assistant", "content": generated_text})
                    return {
                        "generated_text": generated_text,
                        "search_results": search_results
                    }
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")

        # 5. フォールバック対話生成 (Gemini 未設定またはエラー時)
        greeting_keywords = ["あなたは何", "何ができる", "自己紹介", "だれ", "誰", "説明して", "何について", "教えてくれるの"]
        is_greeting = any(k in clean_q for k in greeting_keywords) and len(clean_q) < 20

        if is_greeting:
            generated_text = (
                "私は GlobalLogic Japan の AI 公式アシスタントです！"
                "弊社の会社概要をはじめ、デジタルエンジニアリング、AI（VelocityAI）、ソフトウェア開発、"
                "IT/OTトランスフォーメーション、ならびに各種導入事例や強みについて分かりやすくお答えいたします。"
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
            generated_text = "申し訳ありません。該当する情報が見つかりませんでした。別のキーワードでお尋ねください。"

        history.append({"role": "user", "content": clean_q})
        history.append({"role": "assistant", "content": generated_text})

        return {
            "generated_text": generated_text,
            "search_results": search_results
        }
