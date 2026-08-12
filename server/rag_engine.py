import json
import os
import time
import math
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
        
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.SESSION_TIMEOUT_SECONDS = 180  # 3分無操作で自動初期化

        self.gemini_model = None
        self.init_gemini_llm()

        self.reload_and_reindex()

    def init_gemini_llm(self):
        """Gemini API LLM の初期化"""
        api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key and api_key.strip():
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key.strip())
                for model_candidate in ["gemini-1.5-flash-latest", "gemini-pro", "gemini-1.0-pro"]:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_candidate)
                        logger.info(f"Gemini API LLM ('{model_candidate}') successfully configured.")
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
        if session_id in self.sessions:
            del self.sessions[session_id]
        logger.info(f"[SessionManager] Session '{session_id}' has been reset.")
        return {
            "status": "success",
            "session_id": session_id,
            "message": "ご利用ありがとうございました。セッションを終了・初期化しました。"
        }

    def _get_or_clean_session(self, session_id: str) -> List[Dict[str, str]]:
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
        if os.path.exists(self.qa_json_path):
            current_mtime = os.path.getmtime(self.qa_json_path)
            if self.last_modified_time != 0.0 and current_mtime > self.last_modified_time:
                logger.info(f"Detected update in {self.qa_json_path}. Auto-reindexing Vector DB...")
                self.reload_and_reindex()

    def reload_and_reindex(self) -> Dict[str, Any]:
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

    def _keyword_overlap_score(self, query: str, target_text: str) -> float:
        q_lower = query.lower()
        t_lower = target_text.lower()
        
        keywords = ["サービス", "事業", "概要", "会社", "事例", "強み", "特徴", "提供", "内容", "料金", "費用", "拠点", "オフィス"]
        score = 0.0
        
        for kw in keywords:
            if kw in q_lower and kw in t_lower:
                score += 0.25

        if ("サービス" in q_lower or "事業" in q_lower) and ("主な事業内容" in t_lower or "主なサービス" in t_lower or "概要" in t_lower):
            score += 0.5

        return min(1.0, score)

    def search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        self.check_and_auto_reload()

        if not query or not query.strip():
            return []

        clean_q = query.strip()

        fetch_limit = min(15, len(self.documents)) if self.documents else top_k
        results = self.chroma_collection.query(
            query_texts=[clean_q],
            n_results=fetch_limit
        )

        candidates = []
        if results and "metadatas" in results and len(results["metadatas"]) > 0:
            for i, meta in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                raw_vector_score = max(0.0, 1.0 - float(distance))
                
                calibrated_vector_score = math.pow(max(0.0, (raw_vector_score - 0.70) / 0.30), 2.0)
                kw_score = self._keyword_overlap_score(clean_q, meta["question"] + " " + meta["answer"])
                hybrid_score = (calibrated_vector_score * 0.6) + (kw_score * 0.4)

                candidates.append({
                    "id": meta["qa_id"],
                    "category": meta["category"],
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "score": round(hybrid_score, 4),
                    "raw_score": round(raw_vector_score, 4)
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def is_ambiguous_query(self, query: str, top_score: float) -> bool:
        """質問の意図が不明瞭・曖昧かどうかを判定する"""
        clean_q = query.strip().lower()
        
        very_short_ambiguous = ["詳細", "教えて", "説明", "事例", "費用", "料金", "あれ", "それ", "サービス", "事業", "概要"]
        if clean_q in very_short_ambiguous:
            return True

        if top_score < 0.25:
            return True

        return False

    def generate_rag_response(self, query: str, session_id: str = "default_session", top_k: int = 3) -> Dict[str, Any]:
        """
        マルチターン会話履歴を引き継ぎ、意図確認機能を含む RAG (検索＋生成) 対話応答処理
        """
        clean_q = query.strip()
        if not clean_q:
            return {
                "generated_text": "質問を入力してください。",
                "search_results": []
            }

        history = self._get_or_clean_session(session_id)

        expanded_query = clean_q
        if history and len(clean_q) <= 10:
            last_user_q = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
            if last_user_q:
                expanded_query = f"{last_user_q} {clean_q}"

        search_results = self.search_knowledge_base(query=expanded_query, top_k=top_k)
        top_score = search_results[0]["score"] if search_results else 0.0

        is_ambiguous = self.is_ambiguous_query(clean_q, top_score)

        greeting_keywords = ["あなたは何", "何ができる", "自己紹介", "だれ", "誰", "こんにちは", "はじめまして"]
        is_greeting = any(k in clean_q for k in greeting_keywords) and len(clean_q) < 20

        recent_history = history[-8:] if len(history) > 8 else history
        history_str = "\n".join([
            f"{'ユーザー' if h['role']=='user' else 'AIアシスタント'}: {h['content']}"
            for h in recent_history
        ]) if recent_history else "（これまでの会話はありません）"

        # Gemini LLM 生成処理
        if self.gemini_model:
            try:
                context_str = "\n\n".join([
                    f"[参照ナレッジ {i+1}] (カテゴリ: {r['category']})\nQ: {r['question']}\nA: {r['answer']}"
                    for i, r in enumerate(search_results)
                ])

                system_prompt = f"""あなたは GlobalLogic Japan の公式AIアバターアシスタントです。
これまでの【過去の会話履歴】と【参照ナレッジ】を踏まえ、ユーザーの最新の質問に親しみやすく丁寧で自然に答えてください。

【最重要ルール：意図確認】
ユーザーの質問が「詳細」「教えて」「費用」「事例」など一言で曖昧な場合や、何を質問したいのか対象が不明瞭な場合は、無関係なナレッジの回答を決めつけず、「どのような点についてお知りになりたいでしょうか？（例: 会社概要、サービス内容、導入事例、開発言語など）」と親切に意図を確認・聞き返してください。

【その他の注意事項】
- 質問の意図が「会社全体のサービス内容」であれば、サービス全体（デジタルエンジニアリング、AI、IT/OTトランスフォーメーション等）を包括的に分かりやすく答えてください。
- 直前の会話履歴に指示語（「その会社」など）がある場合は、文脈を考慮して答えてください。

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

        # フォールバック対話生成
        if is_greeting:
            generated_text = (
                "こんにちは！私は GlobalLogic Japan の AI 公式アシスタントです。\n"
                "弊社の会社概要をはじめ、デジタルエンジニアリング、AI（VelocityAI）、ソフトウェア開発、"
                "IT/OTトランスフォーメーション、ならびに各種導入事例や強みについてお答えいたします。"
            )
        elif is_ambiguous:
            generated_text = (
                f"ご質問の「{clean_q}」について、具体的にどのような点をお知りになりたいでしょうか？\n\n"
                "例えば、以下のような内容をご案内できます：\n"
                "・GlobalLogic Japan の会社概要や特徴\n"
                "・提供している主なデジタルエンジニアリングサービス一覧\n"
                "・製造業、通信、金融などの業界別導入事例\n"
                "・AI（VelocityAI）やモダナイゼーションの支援内容\n\n"
                "気になる項目や、より詳しいキーワードをお気軽にお知らせください！"
            )
        elif search_results:
            top = search_results[0]
            if "サービス" in clean_q or "事業" in clean_q or "内容" in clean_q:
                generated_text = (
                    "GlobalLogic Japan は、主に以下のデジタルエンジニアリングサービスを提供しております：\n"
                    "1. エクスペリエンス設計 (UI/UXデザイン)\n"
                    "2. インテリジェンス・エンジニアリング (データ＆AI・VelocityAI)\n"
                    "3. ソフトウェア製品開発 ＆ クラウドプラットフォーム構築\n"
                    "4. IT/OT トランスフォーメーション（製造業・通信・金融向けソリューション）\n\n"
                    f"【詳細回答】\n{top['answer']}"
                )
            elif "事例" in clean_q or "実績" in clean_q:
                generated_text = f"【事例・実績のご紹介】\n{top['answer']}"
            else:
                generated_text = f"ご質問の「{top['question']}」についてお答えいたします。\n\n{top['answer']}"
        else:
            generated_text = (
                f"申し訳ありません。「{clean_q}」に関する直接の該当情報が見つかりませんでした。\n"
                "どのような点についてお知りになりたいか、別のキーワード（例: 会社概要、サービス内容、導入事例など）でお聞かせいただけますでしょうか？"
            )

        history.append({"role": "user", "content": clean_q})
        history.append({"role": "assistant", "content": generated_text})

        return {
            "generated_text": generated_text,
            "search_results": search_results
        }
