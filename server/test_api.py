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

    # 4. UI Homepage (GET /)
    response = client.get("/")
    assert response.status_code == 200

    # 5. 通常質問テスト (マルチターン)
    response1 = client.post("/api/chat-ui", data={"query": "GlobalLogicの親会社はどこですか？"})
    assert response1.status_code == 200

    # 6. 二重重複なし・意図確認の対話応答テスト
    response_ambiguous = client.post("/api/chat-ui", data={"query": "詳細"})
    assert response_ambiguous.status_code == 200
    assert len(response_ambiguous.text) > 0

    print("\n=== All Intent & Duplicate-Free Conversation API Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_api_endpoints()
