# AI音声会話 & RAG対応 3Dアバターシステム バックエンド (FastAPI & RAG) + テストWeb UI

Unreal Engine 5 (UE5) で構築される 3D アバター、および Gemini Live API と連携するための Python (FastAPI) 中継サーバーおよび RAG (検索拡張生成) ナレッジベース環境です。
QA精度を事前に簡易検証するための **HTMX チャット Web UI (テキスト入力 & 音声入力 & 音声読み上げ対応)** も付属しています。

> 📖 **詳細なシステム仕様書および各種手動取り込み手順・運用マニュアルは [MANUAL.md](MANUAL.md) をご覧ください。**

---

## 📁 ディレクトリ構成

```
.
├── globallogic_qa.json       # GlobalLogicに関する101問のQAデータセット (JSON)
├── generate_qa.py            # QAデータセット生成スクリプト
├── requirements.txt          # Python依存ライブラリ
├── MANUAL.md                 # システム仕様書 兼 運用マニュアル
├── server/
│   ├── config.py             # サーバーおよびRAG設定
│   ├── rag_engine.py         # ruri-v3 日本語ベクトル検索エンジン
│   ├── main.py               # FastAPI エンドポイント (REST API / HTMX UI / WebSocket)
│   ├── reindex.py            # ベクトルDB初期化・再取り込みCLIスクリプト
│   ├── templates/
│   │   └── index.html        # HTMX + Web Speech API (音声入力/TTS) チャットUI
│   ├── test_rag.py           # RAG検索単体テスト
│   └── test_api.py           # REST API & HTMX チャット UI の統合テスト
└── README.md                 # 本ドキュメント
```

---

## 📖 QAデータの更新・再取り込み (Re-indexing) 手順

`globallogic_qa.json` を編集・更新した際のベクトル DB（ruri-v3）への再取り込みは以下の方法で可能です。

- **Web UI 画面から (推奨)**: `http://localhost:8000/` の右上にある **「🔄 DB再読み込み」** ボタンをクリック。
- **CLI コマンド**: `C:\Users\right\miniconda3\python.exe server/reindex.py`
- **REST API**: `POST http://localhost:8000/api/reindex`
- **自動検知**: ファイル保存後に検索を実行すると自動で差分検知・再インデックス。

---

## 🚀 起動方法

```powershell
cd C:\Antigravity2\unreal_engine

# 1. 統合テストの実行
C:\Users\right\miniconda3\python.exe server/test_api.py

# 2. FastAPI サーバー & テスト UI の起動
C:\Users\right\miniconda3\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Web UI チャット画面**: `http://localhost:8000/`
- **Swagger API 仕様書**: `http://localhost:8000/docs`
