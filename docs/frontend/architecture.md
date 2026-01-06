# Frontend Architecture: Phoenix Dashboard

The frontend is a single-page application (SPA) built to provide real-time visibility into the ML Platform.

## Tech Stack
*   **Framework**: React 18 + TypeScript (Vite build tool)
*   **Styling**: Tailwind CSS (Utility-first architecture)
*   **State Management**: TanStack Query (React Query) for server-state caching and revalidation.
*   **Icons**: Lucide React.

## Component Design (SOLID)

We adhere to the **Single Responsibility Principle** for UI components:

*   **`StatCard`**: Pure presentational component for displaying metrics.
*   **`mlService`**: Isolated API layer. UI components never call `fetch/axios` directly.
*   **`App.tsx`**: Orchestrator (Container) component that manages layout.

## Key Features
1.  **A/B Test Simulation**:
    *   Buttons to simulate "Good" vs "Bad" customer profiles.
    *   Visual feedback on Model Routing (Champion vs Challenger).
2.  **Drift Visualization**:
    *   Real-time display of Kolmogorov-Smirnov (KS) statistics.
    *   Visual alerts when drift exceeds thresholds.

## Project Structure
```text
frontend/src/
├── api/           # API integration (Axios)
├── types/         # TypeScript Interfaces (DRY)
├── App.tsx        # Main Dashboard Layout
└── main.tsx       # Entry point
```
