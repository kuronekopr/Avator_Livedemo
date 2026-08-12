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

    # 2. Re-index API (POST /api/reindex)
    response = client.post("/api/reindex")
    print("Re-index API Status Code:", response.status_code)
    print("Re-index API Response:", response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 3. UI Homepage (GET /)
    response = client.get("/")
    assert response.status_code == 200

    # 4. HTMX Chat UI Endpoint (POST /api/chat-ui)
    response = client.post("/api/chat-ui", data={"query": "GlobalLogicの親会社はどこですか？"})
    assert response.status_code == 200

    print("\n=== All API & Re-index Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_api_endpoints()
