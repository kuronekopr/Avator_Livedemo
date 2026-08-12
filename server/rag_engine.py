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
        """Gemini API LLM の安定モデル初期化"""
        api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key and api_key.strip():
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key.strip())
                # 安定モデル候補 (1.5-flash / 1.5-pro / gemini-pro)
                candidates = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                for model_candidate in candidates:
                    try:
                        model = genai.GenerativeModel(model_candidate)
                        # テスト生成を行ってエラーが出ないかチェック
                        model.generate_content("test")
                        self.gemini_model = model
                        logger.info(f"Gemini API LLM ('{model_candidate}') successfully configured and verified.")
                        break
                    except Exception as ex:
                        logger.debug(f"Candidate {model_candidate} failed: {ex}")
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
        
        score = 0.0

        # 体制・契約・提供形態に関するキーワードブースト
        if ("体制" in q_lower or "提供" in q_lower or "契約" in q_lower or "アジャイル" in q_lower) and ("体制" in t_lower or "契約" in t_lower or "チーム" in t_lower or "ラボ" in t_lower or "デリバリー" in t_lower):
            score += 0.5

        if "デジタルエンジニアリング" in q_lower and "デジタルエンジニアリング" in t_lower:
            score += 0.4

        keywords = ["ux", "ui", "デザイン", "サービス", "事業", "概要", "会社", "事例", "強み", "特徴", "提供", "内容", "料金", "費用", "拠点", "オフィス"]
        for kw in keywords:
            if kw in q_lower and kw in t_lower:
                score += 0.15

        return min(1.0, score)

    def search_knowledge_base(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        self.check_and_auto_reload()

        if not query or not query.strip():
            return []

        clean_q = query.strip()

        fetch_limit = min(20, len(self.documents)) if self.documents else top_k
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
                hybrid_score = (calibrated_vector_score * 0.5) + (kw_score * 0.5)

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
        """質問の意図が完全に不明瞭か判定（体制・サービス等具体的疑問詞がある場合はFalse）"""
        clean_q = query.strip().lower()
        
        # 明確な疑問詞やトピックが含まれる場合は曖昧判定しない
        clear_topics = ["体制", "方法", "どこ", "いつ", "費用", "料金", "サービス", "契約", "デジタルエンジニアリング"]
        if any(t in clean_q for t in clear_topics):
            return False

        very_short_ambiguous = ["詳細", "教えて", "説明", "あれ", "それ"]
        if clean_q in very_short_ambiguous or clean_q.endswith("詳"):
            return True

        if top_score < 0.20:
            return True

        return False

    def generate_rag_response(self, query: str, session_id: str = "default_session", top_k: int = 4) -> Dict[str, Any]:
        """
        前後の会話文脈を踏まえたマルチターンAIアバター応答生成
        """
        clean_q = query.strip()
        if not clean_q:
            return {
                "generated_text": "質問を入力してくださいね！どのようなことでもお気軽にどうぞ！",
                "search_results": []
            }

        history = self._get_or_clean_session(session_id)

        # 検索用クエリのスマート拡張（マルチターン文脈の引き継ぎ）
        expanded_query = clean_q
        if history:
            last_user_q = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
            if last_user_q and ("体制" in clean_q or "サービス" in clean_q or "内容" in clean_q or "それ" in clean_q or "その" in clean_q):
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

        # Gemini LLM 生成処理（マルチターン対話のコア）
        if self.gemini_model:
            try:
                context_str = "\n\n".join([
                    f"[参照ナレッジ {i+1}] (カテゴリ: {r['category']})\nQ: {r['question']}\nA: {r['answer']}"
                    for i, r in enumerate(search_results)
                ])

                system_prompt = f"""あなたは GlobalLogic Japan の公式3D AIアバターアシスタントです。
ユーザーとリアルタイムでマルチターン（連続した対話）を行っています。以下のルールを厳格に守って回答を生成してください。

【マルチターン対話アバターの最重要ルール】
1. 直前の会話履歴（ユーザーとAIの過去のやり取り）を熟読し、ユーザーが『どのような体制でサービスを提供してくれるのか』『その費用は』など前の発言を受けた質問をしている場合は、直前のトピック（例: デジタルエンジニアリングなど）を前提とした文脈で的確に答えてください。
2. 『どのような体制で〜』という質問に対して、定義（デジタルエンジニアリングとは〜）を繰り返すのは絶対にやめてください。参照ナレッジからグローバルデリバリー体制、ラボ型開発（T&M/専任チーム）、日立グループとの総合力など『体制・提供形態』に関する情報を抽出して説明してください。
3. ユーザーの質問に対して「〇〇に関する体制についてですね！」と親しみやすく相槌を打ち、口語体（アバターの話し言葉）で親切に回答してください。同じ内容を繰り返さないでください。

【過去の会話履歴】
{history_str}

【参照ナレッジ】
{context_str}

【ユーザーの最新の質問】
{clean_q}

【アバターの対話回答文】"""

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

        # フォールバック対話生成 (Gemini未設定または一時エラー時)
        if is_greeting:
            generated_text = (
                "こんにちは！GlobalLogic Japan の AIアバターアシスタントです！😊\n"
                "弊社のデジタルエンジニアリングや UI/UX デザイン、体制や事例などについてお気軽にご質問くださいね。"
            )
        elif "体制" in clean_q or "提供" in clean_q or "チーム" in clean_q:
            # 体制についてのスマートフォールバック合成
            structure_item = next((r for r in search_results if "体制" in r["answer"] or "チーム" in r["answer"] or "契約" in r["answer"] or "ラボ" in r["answer"] or "力" in r["answer"]), search_results[0] if search_results else None)
            answer_text = structure_item["answer"] if structure_item else "グローバル26カ国以上の拠点と専任のエンジニアリングチーム（ラボ型開発・T&M/Fixed Price）を組み、お客様のプロジェクトに柔軟に対応いたします。"
            
            generated_text = (
                "サービス提供体制についてですね！😊\n\n"
                "GlobalLogic では、世界26カ国以上の製品エンジニアリングセンターと日立グループの総合力を活かしたグローバルデリバリー体制を整えております。\n\n"
                f"【具体的な提供形態】\n{answer_text}\n\n"
                "専任チームによるアジャイルなラボ型開発（T&M）や、成果物定義型のプロジェクト契約など、ご要望に合わせた柔軟な体制を提案可能です！"
            )
        elif is_ambiguous:
            generated_text = (
                f"「{clean_q}」についてですね！具体的にどのような点をお知りになりたいでしょうか？\n\n"
                "例えば、会社概要や全体のサービス内容、提供体制、業界別の導入事例などについてお話しできますよ。"
                "お気軽にお知らせくださいね！"
            )
        elif search_results:
            top = search_results[0]
            answer_body = top['answer'].strip()
            generated_text = (
                "お問い合わせいただいた内容についてお話ししますね！\n\n"
                f"{answer_body}\n\n"
                "こちらについて、さらに気になる点や深掘りしたい部分はございますか？"
            )
        else:
            generated_text = (
                f"「{clean_q}」についてのお問い合わせですね。\n"
                "どのような点についてお知りになりたいか、お気軽にお聞かせください！"
            )

        history.append({"role": "user", "content": clean_q})
        history.append({"role": "assistant", "content": generated_text})

        return {
            "generated_text": generated_text,
            "search_results": search_results
        }
