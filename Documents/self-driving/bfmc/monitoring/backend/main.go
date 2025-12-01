package main

import (
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

var (
	upgrader = websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins for now
		},
	}

	clients   = make(map[*websocket.Conn]bool)
	clientsMu sync.Mutex
)

const udpPort = ":8082" // Port for UDP telemetry from agent

func main() {
	// Start UDP listener in a goroutine
	go listenUDP()

	http.HandleFunc("/ws", handleWebSocket)
	// Assuming frontend static files will be in ../frontend/dist
	http.Handle("/", http.FileServer(http.Dir("../frontend/dist")))

	log.Println("Server started on :8080")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}

func listenUDP() {
	addr, err := net.ResolveUDPAddr("udp", udpPort)
	if err != nil {
		log.Fatal("Error resolving UDP address:", err)
	}

	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		log.Fatal("Error listening UDP:", err)
	}
	defer conn.Close()

	log.Printf("Listening for UDP telemetry on %s", udpPort)

	buffer := make([]byte, 4096)
	for {
		n, _, err := conn.ReadFromUDP(buffer)
		if err != nil {
			log.Println("Error reading UDP:", err)
			continue
		}

		telemetryData := buffer[:n]
		log.Printf("Received UDP telemetry: %s", string(telemetryData))
		broadcast(telemetryData)
	}
}

func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println(err)
		return
	}
	defer conn.Close()

	// Add the new client to the map
	clientsMu.Lock()
	clients[conn] = true
	clientsMu.Unlock()

	log.Println("Client connected")

	// Keep the connection alive
	for {
		// Ping pong to keep connection alive
		err := conn.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(time.Second*5))
		if err != nil {
			log.Println("write ping:", err)
			break
		}
		time.Sleep(time.Second * 5)
	}

	// Remove the client from the map
	clientsMu.Lock()
	delete(clients, conn)
	clientsMu.Unlock()
	log.Println("Client disconnected")
}

func broadcast(message []byte) {
	clientsMu.Lock()
	defer clientsMu.Unlock()

	for client := range clients {
		err := client.WriteMessage(websocket.TextMessage, message)
		if err != nil {
			log.Println("write:", err)
			client.Close()
			delete(clients, client)
		}
	}
}
