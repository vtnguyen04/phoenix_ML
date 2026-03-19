# Frontend Architecture: Phoenix Dashboard

## Tech Stack

| Technology | Role |
|-----------|------|
| React 18 + TypeScript | UI framework |
| Vite | Build tool + dev server |
| Vanilla CSS | Styling with custom design system |
| Vitest + React Testing Library | Unit testing (96 tests) |

## Component Architecture (SOLID)

```
frontend/src/
├── api/
│   └── mlService.ts          # API service layer (fetch-based, typed interfaces)
├── components/
│   ├── dashboard/
│   │   ├── StatsGrid.tsx      # Model performance metrics (accuracy, F1, precision, recall)
│   │   ├── DriftPanel.tsx     # Drift monitoring with auto-scan
│   │   ├── ModelInfoCard.tsx   # Champion model metadata display
│   │   ├── GrafanaEmbed.tsx   # Embedded Grafana dashboard (iframe, kiosk mode)
│   │   ├── PipelineStatus.tsx # MLOps pipeline stage visualization
│   │   ├── ServicesStatus.tsx # Infrastructure service health (8 services)
│   │   └── PredictionForm.tsx # Manual prediction input
│   ├── layout/
│   │   ├── Sidebar.tsx        # Navigation with service links
│   │   └── Header.tsx         # App header with auto-refresh badge
│   └── ui/
│       ├── StatCard.tsx       # Reusable metric card
│       └── CustomerSelector.tsx
├── test/                      # 16 test files, 96 tests
├── App.tsx                    # Orchestrator: auto-fetch + 15s refresh interval
├── index.css                  # Design system (dark theme, glassmorphism)
└── main.tsx                   # Entry point
```

## Key Design Decisions

1. **Auto-refresh**: `App.tsx` fetches model info + drift reports on mount with 15-second interval. No manual triggers needed.
2. **Real API data**: `StatsGrid` receives actual metrics (accuracy, F1, precision, recall) from `/api/models/{id}` endpoint.
3. **Grafana embed**: `GrafanaEmbed` uses iframe with `kiosk` mode + dark theme. Requires `GF_SECURITY_ALLOW_EMBEDDING=true` on Grafana.
4. **Pipeline visualization**: `PipelineStatus` derives stage status (completed/active/pending) from real model metadata availability.
5. **No external UI deps**: Vanilla CSS design system with CSS variables — no Tailwind, no component libraries.

## API Integration

| Endpoint | Component | Purpose |
|----------|-----------|---------|
| `GET /api/health` | ServicesStatus | Service health checks |
| `GET /api/models/{id}` | ModelInfoCard, StatsGrid | Model metadata + metrics |
| `GET /api/monitoring/drift/{id}` | DriftPanel | Drift scan results |
| `POST /api/predict` | PredictionForm | Manual prediction |

---
*Updated March 2026*
