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
    print("Session Reset Response:", response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 3. Re-index API (POST /api/reindex)
    response = client.post("/api/reindex")
    print("Re-index API Status Code:", response.status_code)
    assert response.status_code == 200

    # 4. UI Homepage (GET /)
    response = client.get("/")
    assert response.status_code == 200

    # 5. HTMX Multi-turn Chat UI Endpoint (POST /api/chat-ui)
    response1 = client.post("/api/chat-ui", data={"query": "GlobalLogicの親会社はどこですか？"})
    assert response1.status_code == 200

    # 文脈を引き継ぐ連続質問
    response2 = client.post("/api/chat-ui", data={"query": "その会社の設立年は？"})
    assert response2.status_code == 200

    print("\n=== All Multi-turn Session & Reset API Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_api_endpoints()
