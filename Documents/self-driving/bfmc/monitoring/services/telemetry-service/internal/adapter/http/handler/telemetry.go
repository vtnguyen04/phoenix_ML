package handler

import (
    "encoding/json"
    "net/http"
    "telemetry-service/internal/usecase/telemetry"
)

type TelemetryHandler struct {
    queryUseCase *telemetry.QueryUseCase
}

func NewTelemetryHandler(queryUC *telemetry.QueryUseCase) *TelemetryHandler {
    return &TelemetryHandler{queryUseCase: queryUC}
}

func (h *TelemetryHandler) GetLatest(w http.ResponseWriter, r *http.Request) {
    vehicleID := r.URL.Query().Get("vehicle_id")
    if vehicleID == "" {
        http.Error(w, "vehicle_id is required", http.StatusBadRequest)
        return
    }
    
    telemetryData, err := h.queryUseCase.GetLatest(r.Context(), vehicleID)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(telemetryData)
}
