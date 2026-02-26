# WorkOrder Risk Dashboard

GNN-powered risk analysis dashboard for SDG&E work order schedules.

## Architecture

```
WOW-dashboard/
├── app/                    # Next.js App Router (frontend)
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/             # React components
├── lib/                    # Shared TypeScript utilities
├── api/                    # Vercel Python serverless backend
├── vercel.json             # Vercel deployment config
├── package.json
├── requirements.txt        # Python deps
└── tailwind.config.js
```

## Local Development

### Frontend (Next.js)

```bash
cd WOW-dashboard
npm install
npm run dev
```

Frontend runs on http://localhost:3000

### Backend (FastAPI)

```bash
cd WOW-dashboard
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
```

Backend runs on http://localhost:8000

The Next.js dev server proxies `/api/*` requests to the FastAPI backend via the rewrite in `next.config.js`.

### Demo Mode

If model assets (`api/assets/model.pt`, `api/assets/sdge.pt`) are not present, the backend falls back to **demo mode** with heuristic-based predictions. Click "🧪 Run Demo" in the UI to test.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | JSON schedule → risk predictions |
| POST | `/api/predict/csv` | CSV file upload → risk predictions |


## API Assets

To run real GNN inference, place these files in `api/assets/`:

- "graph.yaml" - data config file
- `sdge.pt` — the whole heterogeneous graph
- `model.pt` — trained checkpoint (from `runs/checkpoints/`)
- "feature_schemas/assignemnts.json" - files produced from save_training_schemas.py
- "feature_schemas/districts.json"
- "feature_schemas/engineers.json"
- "feature_schemas/tasks.json"
