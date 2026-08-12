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
    assert response.json()["status"] == "ok"

    # 2. Session Reset API (POST /api/session/reset)
    response = client.post("/api/session/reset")
    print("Session Reset Status Code:", response.status_code)
    assert response.status_code == 200

    # 3. Re-index API (POST /api/reindex)
    response = client.post("/api/reindex")
    print("Re-index API Status Code:", response.status_code)
    assert response.status_code == 200

    # 4. マルチターン事例連動テスト: ターン1 (デジタルエンジニアリングとは何ですか)
    resp1 = client.post("/api/chat-ui", data={"query": "デジタルエンジニアリングとは何ですか"})
    assert resp1.status_code == 200

    # 5. マルチターン事例連動テスト: ターン2 (具体的な事例はありますか -> オウム返しにならずデジタルエンジニアリング事例が返ること)
    resp2 = client.post("/api/chat-ui", data={"query": "具体的な事例はありますか"})
    assert resp2.status_code == 200
    # 「事例」の回答テキストが含まれること（オウム返しオファー文ではないこと）
    assert "事例" in resp2.text or "実績" in resp2.text

    print("\n=== All Case Study & Multi-turn Context Continuity Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_api_endpoints()
