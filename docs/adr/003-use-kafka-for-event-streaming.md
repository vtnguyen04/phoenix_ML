# ADR 003: Asynchronous Event Streaming with Apache Kafka

## Status
Accepted

## Context
A high-throughput inference platform requires zero-latency impact from logging and monitoring tasks. In the initial prototype, synchronous database writes within the FastAPI request cycle increased p99 latency by >20ms. Furthermore, multiple independent services (Drift Monitoring, Historical Audit, and BI Dashboards) require access to the same inference stream. 

We need a distributed, durable, and highly available event bus to decouple the synchronous **Inference Pipeline** from the asynchronous **Observability Pipeline**.

## Decision
We have standardized on **Apache Kafka** as the central event bus for all inference-related events. 

### Implementation Details:
*   **Topic Design**: A primary topic `inference-events` handles the fire-and-forget stream from the API.
*   **Protocol**: We utilize `aiokafka` for non-blocking asynchronous production within Python.
*   **Architecture**: To minimize operational complexity, we utilize **Kafka KRaft Mode** (Metadata Quorum), which removes the dependency on Zookeeper.

## Consequences

### Positive
*   **Inference Latency Reduction**: API response time no longer depends on disk I/O or database performance.
*   **Durability**: Kafka's append-only log ensures that events are safely stored even if the PostgreSQL database or the Drift Detector service is temporarily unavailable.
*   **Extensibility**: We can spin up new consumer groups (e.g., for real-time Fraud Detection) without adding any load to the primary Inference API.
*   **Throughput**: Naturally scales to handle millions of inference requests per day through horizontal partitioning.

### Negative
*   **Operational Overhead**: Requires monitoring Kafka cluster health (ISR, partition offsets, disk usage).
*   **Eventual Consistency**: The monitoring dashboard and logs might lag by a few milliseconds/seconds behind the actual inference.
*   **Payload Management**: Large feature vectors could potentially bloat Kafka topics if not managed (mitigated by only logging compressed or sampled data if necessary).

## Alternatives Considered
*   **RabbitMQ**: Rejected due to lower throughput for large log volumes and lack of persistent replay capabilities.
*   **Redis Pub/Sub**: Rejected due to the lack of message persistence (data is lost if consumers are offline).
*   **AWS Kinesis**: Rejected to maintain cloud-agnostic deployment via Docker/Kubernetes.
