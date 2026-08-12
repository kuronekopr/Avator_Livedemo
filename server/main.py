import os
import json
import html
import logging
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException, WebSocket, WebSocketDisconnect, Form, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import uuid

from server.config import settings
from server.rag_engine import RAGEngine

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("APIServer")

app = FastAPI(
    title="GlobalLogic Avatar & RAG Middleware API",
    description="3D Avatar & Gemini Live API Middleware with RAG Knowledge Search & Multi-turn Session Chat UI",
    version="1.4.0"
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
    
    # RAG (マルチターン会話履歴を引き継いだ検索＋生成) パイプラインを実行
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
    
    # AI応答バブル HTML
    if results:
        top_result = results[0]
        category = html.escape(top_result["category"])
        matched_q = html.escape(top_result["question"])
        score = top_result["score"]
        qa_id = top_result["id"]

        meta_info = f"""
        <div class="qa-meta">
            <span class="tag"><i class="bi bi-folder-fill"></i> {category}</span>
            <span class="score-tag"><i class="bi bi-check-circle-fill"></i> 適合度 Score: {score:.4f} (ID: {qa_id})</span>
            <span style="width: 100%; color: #cbd5e1; font-style: italic; margin-top: 4px;">[参照ナレッジキー] Q: {matched_q}</span>
        </div>
        """
        
        sub_results_html = ""
        if len(results) > 1:
            sub_items = "".join([
                f"<li>[{r['score']:.2f}] <strong>{html.escape(r['question'])}</strong>: {html.escape(r['answer'][:50])}...</li>"
                for r in results[1:]
            ])
            sub_results_html = f"""
            <details style="margin-top: 10px; font-size: 12px; color: #94a3b8; cursor: pointer;">
                <summary style="outline: none;">関連する他のQAナレッジ ({len(results)-1}件)</summary>
                <ul style="margin-top: 6px; padding-left: 16px; display: flex; flex-direction: column; gap: 4px;">
                    {sub_items}
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


# 5. REST API / Healthcheck / WebSocket エンドポイント
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "qa_count": len(rag_engine.documents),
        "chroma_active": rag_engine.is_chroma_active,
        "gemini_active": rag_engine.gemini_model is not None,
        "active_sessions": len(rag_engine.sessions),
        "last_modified_time": rag_engine.last_modified_time
    }

@app.get("/api/search", response_model=SearchResponse)
def search_knowledge_get(
    query: str = Query(..., description="Gemini Live API Function Callから引き渡される検索キーワード"),
    top_k: int = Query(default=3, ge=1, le=10)
):
    results = rag_engine.search_knowledge_base(query=query, top_k=top_k)
    return SearchResponse(query=query, results=results, count=len(results))

@app.post("/api/search", response_model=SearchResponse)
def search_knowledge_post(request: SearchRequest):
    results = rag_engine.search_knowledge_base(query=request.query, top_k=request.top_k)
    return SearchResponse(query=request.query, results=results, count=len(results))

@app.websocket("/ws/avatar")
async def websocket_avatar_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket Client connected to Avatar endpoint.")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "search_request":
                q = message.get("query", "")
                sess_id = message.get("session_id", "default_session")
                rag_out = rag_engine.generate_rag_response(query=q, session_id=sess_id, top_k=3)
                await websocket.send_text(json.dumps({"type": "search_response", "generated_text": rag_out["generated_text"], "results": rag_out["search_results"]}, ensure_ascii=False))
            else:
                await websocket.send_text(json.dumps({"type": "echo", "message": f"Received: {message}"}))
    except WebSocketDisconnect:
        logger.info("WebSocket Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")

if __name__ == "__main__":
    uvicorn.run("server.main:app", host=settings.HOST, port=settings.PORT, reload=True)
