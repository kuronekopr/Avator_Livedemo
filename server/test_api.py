import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from server.main import app

def test_api_endpoints():
    client = TestClient(app)
    
    # 1. Health check
    response = client.get("/health")
    print("Health Check Status Code:", response.status_code)
    assert response.status_code == 200

    # 2. Session Reset API (POST /api/session/reset)
    response = client.post("/api/session/reset")
    print("Session Reset Status Code:", response.status_code)
    assert response.status_code == 200

    # 3. 画面キャプチャの再現シナリオテスト:
    # ターン1: 「会社の事業内容について教えて」
    resp1 = client.post("/api/chat-ui", data={"query": "会社の事業内容について教えて"})
    assert resp1.status_code == 200
    print("\n--- Turn 1 Response ---")
    assert "GlobalLogic" in resp1.text or "事業" in resp1.text

    # ターン2 (途中割り込み): 「デジタルエンジニアリングとは」
    # 直前の「会社の事業内容〜」と無理に結合されず、ID: 102 (デジタルエンジニアリングの定義) が独立して取得されること
    resp2 = client.post("/api/chat-ui", data={"query": "デジタルエンジニアリングとは"})
    assert resp2.status_code == 200
    print("\n--- Turn 2 Response ---")
    
    # 1回目の事業内容(ID:4)のオウム返しではなく、デジタルエンジニアリングの定義が返っていること
    assert "デザイン思考" in resp2.text or "技術支援" in resp2.text or "最先端のソフトウェア開発" in resp2.text

    print("\n=== Independent Query & Interruption Test Passed Successfully! ===")

if __name__ == "__main__":
    test_api_endpoints()
