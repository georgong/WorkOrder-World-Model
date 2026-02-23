# Capstone Dashboard Deployment on Vercel — Complete Implementation Plan

## 1) Goal and Scope
Deploy an interactive web dashboard that:
- Lets users **upload a schedule** (CSV/JSON)
- Backend **reconstructs a heterogeneous graph**
- Runs your **trained GNN risk model**
- Returns **assignment-level risks + schedule-level metrics**
- Visualizes results (tables, charts, optional graph view) through an dashboards

Deployment target:
- **Vercel** for hosting + serverless backend
- GitHub repo as the single source of truth

---

## 2) Recommended Architecture (Vercel-Friendly)

### Frontend (UI)
- **Next.js (TypeScript)** hosted on Vercel
- Pages:
  - Upload schedule
  - Results dashboard (risk score, workload imbalance, congestion indicators)
  - Drill-down table for risky assignments
  - Export/report view

### Backend (Inference API)

Two viable patterns — pick one:

#### A. All-in-Vercel (Recommended if inference is lightweight)
- Backend runs as **FastAPI on Vercel Python Runtime** (serverless).
- Pros: single deployment platform, clean integration.
- Cons: function runtime limits (time/memory), cold-start latency.

#### B. Split Backend (Recommended if inference is heavy)
- Frontend on Vercel.
- Backend deployed separately (Cloud Run / Render / AWS ECS / lab server).
- Vercel frontend calls external inference API.
- Pros: fewer constraints, better for PyTorch Geometric.
- Cons: 2 deployments to manage.

Start with A first.

---

## 3) Tech Stack

- Frontend: Next.js + TypeScript + React
- Backend: FastAPI (Python)
- Model runtime: PyTorch (+ PyG if required)
- Data parsing: Pandas / Polars
- Visualization: Recharts / Plotly / custom table components

---

## 4) Repository Layout (Monorepo)

```
capstone-dashboard/
  app/                        # Next.js App Router (or pages/)
    page.tsx
    dashboard/page.tsx
    api-client.ts
  components/
    UploadPanel.tsx
    MetricsCards.tsx
    RiskTable.tsx
    Charts.tsx
    ErrorBanner.tsx
  public/
  api/                        # Vercel Python serverless
    index.py                  # FastAPI entrypoint
    inference/
      schema.py
      loader.py
      graph_builder.py
      model.py
      postprocess.py
    assets/
      graph.yaml
      model.pt
  requirements.txt
  package.json
  vercel.json
  README.md
```

---

## 5) Key Design Decisions

### 5.1 Input Format

Option 1: JSON (recommended for API integration)

Option 2: CSV upload (convert to canonical JSON in backend)

Recommended: Accept CSV upload → convert to canonical JSON internally.

---

### 5.2 Output Format

Standardized JSON response:

- Schedule-level metrics
- Per-assignment risk list
- Chart-ready arrays

---

## 6) API Contract

### 6.1 Health Check
GET /api/health

Response:
```
{ "status": "ok" }
```

---

### 6.2 Predict Endpoint

POST /api/predict

Request (JSON):
```
{
  "schedule_id": "optional",
  "records": [
    {
      "assignment_id": "...",
      "task_id": "...",
      "engineer_id": "...",
      "district": "...",
      "start_time": "...",
      ...
    }
  ]
}
```

Response:
```
{
  "schedule_metrics": {
    "overall_risk_score": 0.73,
    "expected_overdue_rate": 0.12,
    "workload_imbalance_score": 0.31,
    "congestion_score": 0.28
  },
  "assignment_predictions": [
    {
      "assignment_id": "A123",
      "pred_completion_hours": 2.3,
      "risk_score": 0.81,
      "top_factors": ["district congestion", "tight slack time"]
    }
  ],
  "charts": {
    "risk_histogram": [...],
    "risk_by_district": [...],
    "workload_by_engineer": [...]
  },
  "metadata": {
    "runtime_ms": 842,
    "model_version": "v1"
  }
}
```

---

### 6.3 Upload to Blob (Optional)

POST /api/upload-url

Returns signed upload URL.

Frontend uploads directly, then calls `/api/predict` with file URL.

---

## 7) Backend Implementation (FastAPI)

### 7.1 Model Loading
- Load model once at startup.
- Cache model and config globally.

### 7.2 Inference Flow
1. Parse input
2. Validate schema
3. Build heterogeneous graph
4. Run model inference
5. Aggregate schedule-level metrics
6. Return JSON payload

### 7.3 Error Handling
- 400 for bad input
- 500 for runtime/model issues
- Include request_id for debugging

---

## 8) Frontend Implementation (Next.js)

### Pages
- `/` Upload page
- `/dashboard` Results page

### UI Requirements
- Upload progress state
- Processing indicator
- Error handling display
- Export results (JSON/CSV)
- Optional HTML snapshot export

---

## 9) Deployment Steps

1. Push to GitHub.
2. Import repo in Vercel.
3. Framework preset: Next.js.
4. Ensure Python API detected.
5. Add environment variables:
   - MODEL_PATH
   - BLOB_READ_WRITE_TOKEN
6. Deploy.
7. Test health + prediction endpoint.

---

## 10) Required Files

Required:
- package.json
- api/index.py
- requirements.txt
- README.md
- Sample schedule file

Optional:
- model.pt
- graph.yaml
- vercel.json
- tests/

---

## 11) Testing Plan

Unit Tests:
- Schedule parser
- Graph builder
- Metric aggregation

Integration Tests:
- /api/predict endpoint with sample input

Regression Tests:
- Fixed seed → consistent outputs

---

## 12) Performance Considerations

If inference is slow:
- Reduce graph size
- Cache static embeddings
- Precompute features
- Consider moving backend externally

If timeouts occur:
- Async job pattern
- Split backend architecture


---

## 14) Final Deliverables

- Public Vercel URL
- GitHub repo link
- Dashboard screenshots
- Architecture diagram (schedule → graph → model → dashboard)
- Demo-ready example schedule