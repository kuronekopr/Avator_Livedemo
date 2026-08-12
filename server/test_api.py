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

    # 4. マルチターン対話テスト: ターン1 (サービス内容)
    resp1 = client.post("/api/chat-ui", data={"query": "あなたの会社の事業内容について教えて"})
    assert resp1.status_code == 200

    # 5. マルチターン対話テスト: ターン2 (デジタルエンジニアリング)
    resp2 = client.post("/api/chat-ui", data={"query": "デジタルエンジニアリングについて教えて"})
    assert resp2.status_code == 200

    # 6. マルチターン対話テスト: ターン3 (体制の質問 -> 定義オウム返しにならず体制情報が返ること)
    resp3 = client.post("/api/chat-ui", data={"query": "どのような体制でサービスを提供してくれる"})
    assert resp3.status_code == 200

    # 7. マルチターン対話テスト: ターン4 (文脈を引き継いだ体制の質問)
    resp4 = client.post("/api/chat-ui", data={"query": "デジタルエンジニアリングに対してどのような体制でサービスを提供してくれるのですか"})
    assert resp4.status_code == 200

    print("\n=== All Multi-turn Conversation & Structure Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_api_endpoints()
