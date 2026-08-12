import json
import os
import uuid
import logging
import html
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Form, Cookie, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.config import settings
from server.rag_engine import RAGEngine

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MainServer")

app = FastAPI(
    title="GlobalLogic RAG Avatar Chat Service",
    version="1.0.0",
    description="ChromaDB + ruri-v3 と Gemini LLM によるマルチターン対話アバター中継 API サーバー"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAGエンジンの初期化
rag_engine = RAGEngine(qa_json_path=settings.QA_JSON_PATH)


# Pydantic モデル定義
class SearchRequest(BaseModel):
    query: str = Field(..., example="GlobalLogicの親会社はどこですか？", description="検索クエリ")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="取得件数")

class SearchResultItem(BaseModel):
    id: int
    category: str
    question: str
    answer: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    count: int

class SessionResetRequest(BaseModel):
    session_id: Optional[str] = None


# 1. UIトップページ (HTML)
@app.get("/", response_class=HTMLResponse)
def read_index_ui(response: Response, session_id: Optional[str] = Cookie(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)
        
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template file index.html not found.")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, headers={"Set-Cookie": f"session_id={session_id}; Path=/"})


# 2. Vector DB 初期化・再取り込み API (POST /api/reindex)
@app.post("/api/reindex")
def reindex_vector_db():
    try:
        result = rag_engine.reload_and_reindex()
        return result
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")


# 3. セッション初期化 API (POST /api/session/reset)
@app.post("/api/session/reset")
def reset_chat_session(
    request: Optional[SessionResetRequest] = None,
    session_id_cookie: Optional[str] = Cookie(None, alias="session_id")
):
    target_session_id = (request and request.session_id) or session_id_cookie or "default_session"
    result = rag_engine.reset_session(session_id=target_session_id)
    return result


# 4. HTMX専用 RAG マルチターンチャット処理エンドポイント
@app.post("/api/chat-ui", response_class=HTMLResponse)
def handle_chat_ui(
    query: str = Form(...),
    session_id_cookie: Optional[str] = Cookie(None, alias="session_id"),
    session_id_form: Optional[str] = Form(None)
):
    active_session_id = session_id_form or session_id_cookie or "default_session"
    escaped_query = html.escape(query.strip())
    
    # RAG (マルチターン会話履歴を引き継いだアバター対話検索＋生成) パイプラインを実行
    rag_output = rag_engine.generate_rag_response(query=query, session_id=active_session_id, top_k=3)
    generated_text = html.escape(rag_output["generated_text"]).replace("\n", "<br>")
    results = rag_output["search_results"]

    # ユーザー発言バブル HTML
    user_html = f"""
    <div class="message-row user">
        <div class="avatar"><i class="bi bi-person-fill"></i></div>
        <div class="bubble">{escaped_query}</div>
    </div>
    """
    
    # AI応答バブル HTML (アバター対話重視のUIデザイン)
    if results:
        top_result = results[0]
        category = html.escape(top_result["category"])
        matched_q = html.escape(top_result["question"])
        score = top_result["score"]
        qa_id = top_result["id"]

        meta_info = f"""
        <div class="qa-meta" style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.08);">
            <span class="tag"><i class="bi bi-folder-fill"></i> {category}</span>
            <span class="score-tag"><i class="bi bi-check-circle-fill"></i> 関連性: {score:.4f} (ID: {qa_id})</span>
        </div>
        """
        
        sub_results_html = f"""
        <details style="margin-top: 8px; font-size: 11px; color: #94a3b8; cursor: pointer;">
            <summary style="outline: none;">参照したナレッジソース ({len(results)}件)</summary>
            <ul style="margin-top: 6px; padding-left: 16px; display: flex; flex-direction: column; gap: 4px;">
                <li>Q: <strong>{matched_q}</strong></li>
                {"".join([f"<li>Q: {html.escape(r['question'])}</li>" for r in results[1:]])}
            </ul>
        </details>
        """

        ai_html = f"""
        <div class="message-row ai">
            <div class="avatar"><i class="bi bi-robot"></i></div>
            <div class="bubble">
                <div class="answer-text">{generated_text}</div>
                {meta_info}
                {sub_results_html}
            </div>
        </div>
        """
    else:
        ai_html = f"""
        <div class="message-row ai">
            <div class="avatar"><i class="bi bi-robot"></i></div>
            <div class="bubble">
                <div class="answer-text">{generated_text}</div>
            </div>
        </div>
        """

    return HTMLResponse(content=user_html + ai_html)


# 5. ナレッジベース検索 API (GET /api/search - Gemini Live API Tool Call 等で利用)
@app.get("/api/search", response_model=SearchResponse)
def search_api(query: str, top_k: int = 3):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    results = rag_engine.search_knowledge_base(query=query, top_k=top_k)
    return SearchResponse(
        query=query,
        results=[
            SearchResultItem(
                id=r["id"],
                category=r["category"],
                question=r["question"],
                answer=r["answer"],
                score=r["score"]
            )
            for r in results
        ],
        count=len(results)
    )


# 6. ヘルスチェック API (GET /health)
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "vector_db_active": rag_engine.is_chroma_active,
        "loaded_documents_count": len(rag_engine.documents),
        "active_sessions_count": len(rag_engine.sessions),
        "embedding_model": "cl-nagoya/ruri-v3-310m",
        "gemini_llm_active": rag_engine.gemini_model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
