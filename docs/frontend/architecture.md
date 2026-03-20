# Frontend Architecture: Phoenix Dashboard

## Tech Stack

| Technology | Role |
|-----------|------|
| React 18 + TypeScript | UI framework |
| Vite | Build tool + dev server |
| Vanilla CSS | Styling with custom design system (dark theme, glassmorphism) |
| Vitest + React Testing Library | Unit testing (96 tests, 16 test files) |

## Component Architecture (SOLID)

```
frontend/src/
├── api/
│   └── mlService.ts          # API service layer (fetch-based, typed interfaces)
├── components/
│   ├── dashboard/
│   │   ├── StatsGrid.tsx      # Model performance metrics (accuracy, F1, precision, recall)
│   │   ├── DriftPanel.tsx     # Drift monitoring with auto-scan + KS/PSI/Chi²/Wasserstein
│   │   ├── ModelInfoCard.tsx   # Champion model metadata display
│   │   ├── GrafanaEmbed.tsx   # Embedded Grafana dashboard (iframe, kiosk mode)
│   │   ├── PipelineStatus.tsx # MLOps pipeline stage visualization
│   │   ├── ServicesStatus.tsx # Infrastructure service health (14+ services)
│   │   └── PredictionForm.tsx # Manual prediction input form
│   ├── layout/
│   │   ├── Sidebar.tsx        # Navigation with service links (API, Grafana, MLflow, etc.)
│   │   └── Header.tsx         # App header with auto-refresh badge
│   └── ui/
│       ├── StatCard.tsx       # Reusable metric card
│       └── CustomerSelector.tsx
├── test/                      # 16 test files, 96 tests
├── App.tsx                    # Orchestrator: auto-fetch + 15s refresh interval
├── index.css                  # Design system (dark theme, glassmorphism, CSS variables)
└── main.tsx                   # Entry point
```

## Key Design Decisions

1. **Auto-refresh**: `App.tsx` fetches model info + drift reports on mount with 15-second interval. No manual triggers needed.
2. **Real API data**: `StatsGrid` receives actual metrics (accuracy, F1, precision, recall) from `/models/{id}` endpoint. Model-agnostic — works with any model in the registry.
3. **Grafana embed**: `GrafanaEmbed` uses iframe with `kiosk` mode + dark theme. Requires `GF_SECURITY_ALLOW_EMBEDDING=true` on Grafana service.
4. **Pipeline visualization**: `PipelineStatus` derives stage status (completed/active/pending) from real model metadata availability.
5. **No external UI deps**: Vanilla CSS design system with CSS variables — no Tailwind, no component libraries. Fully custom dark theme with glassmorphism effects.
6. **Model selector**: Dashboard supports switching between models (credit-risk, house-price, fraud-detection, image-class).

## API Integration

All endpoints are prefixed with the base API URL (default: `http://localhost:8001`). **No `/api/` prefix** — routes are mounted directly at root.

| Endpoint | Component | Purpose |
|----------|-----------|---------|
| `GET /health` | ServicesStatus | Service health checks |
| `GET /models/{id}` | ModelInfoCard, StatsGrid | Model metadata + metrics |
| `GET /monitoring/drift/{id}` | DriftPanel | Drift scan results |
| `GET /monitoring/reports/{id}` | DriftPanel | Historical drift reports |
| `POST /predict` | PredictionForm | Manual prediction |
| `POST /predict/batch` | PredictionForm | Batch prediction |

## Environment

```bash
# Development
cd frontend && npm install && npm run dev  # http://localhost:5173

# Docker
# Runs at http://localhost:5174 (port mapping 5174→5173)
# API URL configured via env: VITE_API_URL

# Testing
npx vitest run          # 96 tests
npx vitest run --ui     # Interactive UI
```

---
*Updated March 2026*
