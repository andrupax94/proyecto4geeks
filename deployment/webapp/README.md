# DataScience Multifunction Scaffold

Monorepo base:
- `frontend/` → Next.js + TypeScript
- `backend/` → FastAPI

## What is included
- 70/30 main layout with a collapsible sidebar
- Wiki section
- EDA dashboard section
- Model training dashboard section
- SoundList section
- TestSound section
- Mobile view panel on the right
- FastAPI stubs with basic responses

## Start

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
uvicorn app.main:app --reload
```
