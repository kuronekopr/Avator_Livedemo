# AI音声会話＆RAG対応 3Dアバターシステム バックエンド仕様書 兼 運用マニュアル

本書は、Unreal Engine 5 (UE5) 3D アバターおよび Gemini Live API と連携する **FastAPI 中継サーバー**、および **日本語 Embedding モデル (ruri-v3) を搭載した Vector DB (RAG) システム** の基本仕様および運用マニュアルです。

---

## 1. システム仕様概要

### 1.1 全体アーキテクチャ
本システムは、UE5 クライアントと Gemini Live API の間に位置し、セキュリティの確保、知識検索 (RAG)、および QA 精度の検証環境を提供します。

```
[ UE5 クライアント (MacBook Air M4) ]
   ├── マイク音声入力 (PCM 16kHz) ──(WebSocket)──> [ 中継サーバー (FastAPI/Python) ] ──(WebSocket)──> [ Gemini Live API ]
   ├── スピーカー音声再生 (PCM 24kHz) <──(WebSocket)──┤  ├── RAG Tool Call (search_knowledge_base)
   └── Lip Sync (OVRLipSync/BlendShape) └── [ Vector DB: ChromaDB (ruri-v3) ]
```

### 1.2 環境変数・Gemini API Key の設定 (`.env`)
Gemini API を使用して高度な対話文章を生成する場合、プロジェクトルート直下の **`.env`** ファイルに API キーを記述します。

```ini
# .env ファイルの例
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here...
```

> 🔒 **セキュリティ上の注意**: `.env` ファイルは Git 管理対象外 (`.gitignore`) に登録されているため、Public リポジトリへ秘密鍵が漏洩することはありません。ひな形としては `.env.example` をご参照ください。

---

## 2. QAデータセット仕様 (`globallogic_qa.json`)

### 2.1 データスキーマ
`globallogic_qa.json` は、以下の JSON 配列形式で管理されます。

```json
[
  {
    "id": 1,
    "Category": "会社概要・基本情報",
    "Question": "GlobalLogicとはどのような会社ですか？",
    "Answer": "GlobalLogic（グローバルロジック）は、日立グループのデジタルエンジニアリング企業であり..."
  }
]
```

---

## 3. 📖 マニュアル：QAデータの更新・再取り込み手順

`globallogic_qa.json` に新しい質問を追加・修正した際、ベクトルデータベース（ruri-v3 ChromaDB）へ**再取り込み（初期化＆再インデックス）**を行う手順は以下の **4 通り** あります。

- **方法 A (推奨)**: ブラウザで `http://localhost:8000/` にアクセスし、右上ヘッダーの **「🔄 DB再読み込み」** ボタンをクリック。
- **方法 B (CLI)**: ターミナルで `C:\Users\right\miniconda3\python.exe server/reindex.py` を実行。
- **方法 C (API)**: `POST http://localhost:8000/api/reindex` を呼び出し。
- **方法 D (自動検知)**: `.json` ファイル保存直後の検索実行時に更新日時を自動検知して再取り込み。

---

## 4. API エンドポイント一覧

| メソッド | パス | 説明 |
| :--- | :--- | :--- |
| `GET` | `/` | QA 精度検証用 HTMX チャット Web UI 画面 |
| `POST` | `/api/chat-ui` | HTMX 専用 RAG チャットレスポンス API (LLM Conversational) |
| `GET` | `/api/search` | Gemini Live API Tool Call (`search_knowledge_base`) 用 API |
| `POST` | `/api/search` | POST 形式の検索 API |
| `POST` | `/api/reindex` | ベクトル DB の初期化・再インデックス API |
| `GET` | `/health` | サーバー状態、Gemini 連携有無、QA 件数ヘルスチェック |
| `WS` | `/ws/avatar` | UE5 3D アバター用 WebSocket ストリーミング接続 |

---

## 5. サーバー起動方法

```powershell
cd C:\Antigravity2\unreal_engine
C:\Users\right\miniconda3\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```
