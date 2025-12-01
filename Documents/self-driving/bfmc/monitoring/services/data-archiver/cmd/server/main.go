package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	// "data-archiver/internal/di"
	// "data-archiver/internal/infrastructure/config"
)

func main() {
	// Load configuration
	// cfg, err := config.Load()
	// if err != nil {
	// 	log.Fatalf("Failed to load config: %v", err)
	// }

	// Build dependency container
	// container, cleanup, err := di.BuildContainer(cfg)
	// if err != nil {
	// 	log.Fatalf("Failed to build container: %v", err)
	// }
	// defer cleanup()

	// Start services
	// ctx, cancel := context.WithCancel(context.Background())
	// defer cancel()

	// go container.MessageConsumer.Start(ctx)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down gracefully...")
}
