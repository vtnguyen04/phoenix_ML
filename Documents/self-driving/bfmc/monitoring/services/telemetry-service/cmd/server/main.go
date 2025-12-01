package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"telemetry-service/internal/di"
	"telemetry-service/internal/infrastructure/config"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Build dependency container
	container, cleanup, err := di.BuildContainer(cfg)
	if err != nil {
		log.Fatalf("Failed to build container: %v", err)
	}
	defer cleanup()

	// Start services
	// ctx, cancel := context.WithCancel(context.Background())
	// defer cancel()

	// Start HTTP server
	go func() {
		log.Println("HTTP server listening on :8080")
		if err := container.HTTPServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server ListenAndServe: %v", err)
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down gracefully...")

	// Shutdown the server
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := container.HTTPServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP server Shutdown: %v", err)
	}
}

