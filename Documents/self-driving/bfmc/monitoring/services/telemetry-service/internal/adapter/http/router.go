package http

import (
	"net/http"
	"telemetry-service/internal/adapter/http/handler"
)

func NewRouter(telemetryHandler *handler.TelemetryHandler) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/telemetry/latest", telemetryHandler.GetLatest)
	return mux
}
