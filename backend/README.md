# SuperAgent Backend (M1)

FastAPI backend: OpenAI-compatible streaming chat over WebSocket.

## Setup (Windows, Python 3.10+)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `backend\.env.example` to `backend\.env` and fill in `OPENAI_API_KEY`.

## Run the API

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload
```

- `GET /api/health` → `{"status":"ok"}`
- `WS  /ws/chat`     → send `{"type":"user_message","request_id":"r1","content":"..."}`

## Test

```powershell
.venv\Scripts\python.exe -m pytest backend\tests -v
```
