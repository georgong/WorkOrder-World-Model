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
│   ├── UploadPanel.tsx     # CSV upload + drag & drop
│   ├── Dashboard.tsx       # Tab navigation + layout
│   ├── MetricsCards.tsx    # Schedule-level risk metrics
│   ├── RiskTable.tsx       # Sortable assignment risk table
│   └── Charts.tsx          # Recharts visualizations
├── lib/                    # Shared TypeScript utilities
│   ├── types.ts            # Type definitions
│   └── api-client.ts       # API client functions
├── api/                    # Vercel Python serverless backend
│   ├── index.py            # FastAPI entrypoint
│   └── inference/
│       ├── model.py        # HeteroSAGERegressor model
│       └── schema.py       # Data schemas + column aliases
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

## Deploy to Vercel

1. Push this folder to GitHub
2. Import repo in [Vercel](https://vercel.com)
3. Set root directory to `WOW-dashboard`
4. Framework preset: **Next.js**
5. Add model assets to `api/assets/` (optional — demo mode works without them)
6. Deploy

## Model Assets (Optional)

To run real GNN inference, place these files in `api/assets/`:

- `sdge.pt` — the heterogeneous graph
- `model.pt` — trained checkpoint (from `runs/checkpoints/`)
