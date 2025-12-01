package di

import (
	"net/http"
	"telemetry-service/internal/adapter/http/handler"
	"telemetry-service/internal/usecase/telemetry"
	"telemetry-service/internal/adapter/repository/mock"
	httpt "telemetry-service/internal/adapter/http"
)

type Container struct {
	HTTPServer *http.Server
}

func BuildContainer(cfg interface{}) (*Container, func(), error) {
	// repository
	repo := mock.NewTelemetryRepository()

	// usecase
	queryUseCase := telemetry.NewQueryUseCase(repo)

	// handler
	telemetryHandler := handler.NewTelemetryHandler(queryUseCase)

	// router
	router := httpt.NewRouter(telemetryHandler)

	server := &http.Server{
		Addr:    ":8080",
		Handler: router,
	}

	return &Container{
		HTTPServer: server,
	}, func() {}, nil
}
