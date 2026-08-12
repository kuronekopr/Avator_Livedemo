# AI音声会話＆RAG対応 3Dアバターシステム バックエンド仕様書 兼 運用マニュアル

本書は、Unreal Engine 5 (UE5) 3D アバターおよび Gemini Live API と連携する **FastAPI 中継サーバー**、および **マルチターン会話記憶対応 RAG (ChromaDB + ruri-v3) システム** の仕様書および運用マニュアルです。

---

## 1. システム仕様概要

### 1.1 全体アーキテクチャ
```
[ UE5 クライアント (MacBook Air M4) ]
   ├── マイク音声入力 (PCM 16kHz) ──(WebSocket)──> [ 中継サーバー (FastAPI/Python) ] ──(WebSocket)──> [ Gemini Live API ]
   ├── スピーカー音声再生 (PCM 24kHz) <──(WebSocket)──┤  ├── Multi-turn Session Manager (3分無操作タイムアウト)
   └── Lip Sync (OVRLipSync/BlendShape) └── [ Vector DB: ChromaDB (ruri-v3) + 意図確認ガイド ]
```

### 1.2 マルチターン対話 ＆ 質問の意図確認仕様
- **マルチターン対話 (会話文脈の保持)**:
  - ユーザーが「親会社はどこですか？」「その会社の設立年は？」と連続して質問した場合、直前の会話文脈（指示語「その会社」など）を理解して応答します。
- **❓ 質問の意図確認（聞き返しロジック）**:
  - ユーザーの入力が「詳細」「費用」「事例」など一言で曖昧な場合や、検索適合スコアが一定未満で対象が特定できない場合、**無関係なドキュメントの回答を決めつけず、「どのような点についてお知りになりたいでしょうか？（例: 会社概要、サービス内容、導入事例など）」とチャット画面上で親切に意図を確認・聞き返します。**
- **(要件A) 3分無操作自動タイムアウト**:
  - 会話が入力されてから 3 分間（180秒）次の質問がない場合、チャット画面に「**ご利用ありがとうございました。**」と出力してセッションをタイムアウト終了し、会話履歴を初期化します。
- **(要件B) UI「🧹 セッション初期化」ボタン**:
  - チャット UI ヘッダーに配置。クリックすると即座に「**ご利用ありがとうございました。**」と出力し、対話セッションおよび会話履歴を完全リセット初期化します。

---

## 2. 📖 マニュアル：QAデータの更新・再取り込み手順

- **方法 A (推奨)**: チャット UI (`http://localhost:8000/`) 右上の **「🔄 DB再読み込み」** ボタンをクリック。
- **方法 B (CLI)**: `C:\Users\right\miniconda3\python.exe server/reindex.py` を実行。
- **方法 C (API)**: `POST http://localhost:8000/api/reindex` を実行。
- **方法 D (自動検知)**: `.json` 保存後の検索実行時に自動更新検知。

---

## 3. API エンドポイント一覧

| メソッド | パス | 説明 |
| :--- | :--- | :--- |
| `GET` | `/` | マルチターン対話 ＆ 意図確認機能対応 HTMX チャット Web UI |
| `POST` | `/api/chat-ui` | HTMX 専用マルチターン RAG チャット応答 API |
| `POST` | `/api/session/reset` | **対話セッション・会話履歴の完全初期化 API** |
| `POST` | `/api/reindex` | ベクトル DB の初期化・再インデックス API |
| `GET` | `/api/search` | Gemini Live API Tool Call (`search_knowledge_base`) 用 API |
| `GET` | `/health` | サーバー状態・アクティブセッション数ヘルスチェック |
| `WS` | `/ws/avatar` | UE5 3D アバター用 WebSocket ストリーミング接続 |

---

## 4. サーバー起動方法

```powershell
cd C:\Antigravity2\unreal_engine
C:\Users\right\miniconda3\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```
