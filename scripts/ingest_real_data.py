import asyncio
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.domain.inference.entities.model import Model
from src.infrastructure.persistence.postgres_model_registry import (
    PostgresModelRegistry,
)
from src.shared.ingestion.redis_ingestor import RedisDataIngestor
from src.shared.ingestion.service import IngestionService

SUCCESS_STATUS = 200


async def main() -> None:
    print("🚀 Starting full-loop data ingestion (Features + Ground Truth)...")

    # 1. Setup Feature Store Ingestor (Redis)
    # Using localhost:6380 as mapped in compose.yaml
    ingestor = RedisDataIngestor(redis_url="redis://localhost:6380")
    service = IngestionService(ingestor)

    test_data = [
        {
            "entity_id": "cust-201",
            "data": {"f1": 1.0, "f2": 1.0, "f3": 1.0, "f4": 1.0},
            "actual": 1,
        },
        {
            "entity_id": "cust-202",
            "data": {"f1": -1.0, "f2": -1.0, "f3": -1.0, "f4": -1.0},
            "actual": 0,
        },
    ]

    await service.process_batch(test_data)
    print("✅ Ingested features into Redis.")

    # 2. Register Model in Docker Postgres (Port 5433)
    db_url = "postgresql+asyncpg://user:pass@localhost:5433/phoenix"
    engine = create_async_engine(db_url)
    session_factory = async_sessionmaker(bind=engine)

    async with session_factory() as session:
        registry = PostgresModelRegistry(session)
        model_uri = "local:///models/credit_risk/v1/model.onnx"
        credit_model = Model(
            id="credit-risk",
            version="v1",
            uri=model_uri,
            framework="onnx",
            metadata={"role": "champion", "metrics": {"f1_score": 0.8}},
            created_at=datetime.now(UTC),
            is_active=True,
        )
        await registry.save(credit_model)
        await registry.update_stage("credit-risk", "v1", "champion")
        print("✅ Model registered.")

    # 3. Predict and then Give Feedback (Ground Truth)
    print("\n🔮 Predicting and Collecting Feedback...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        for item in test_data:
            eid = item["entity_id"]
            # 3.1 Get Prediction
            resp = await client.post(
                "http://localhost:8001/predict",
                json={"model_id": "credit-risk", "entity_id": eid},
            )
            if resp.status_code == SUCCESS_STATUS:
                pred_data = resp.json()
                pid = pred_data["prediction_id"]
                print(
                    f"✔️ Predicted for {eid}: {pred_data['result']} "
                    f"(ID: {pid[:8]}...)"
                )

                # 3.2 Send Ground Truth (Feedback)
                fb_resp = await client.post(
                    "http://localhost:8001/feedback",
                    json={"prediction_id": pid, "ground_truth": item["actual"]},
                )
                if fb_resp.status_code == SUCCESS_STATUS:
                    print(
                        f"   ✨ Feedback collected for {pid[:8]}... "
                        f"(Actual: {item['actual']})"
                    )
                else:
                    print(f"   ❌ Feedback failed: {fb_resp.text}")
            else:
                print(f"❌ Prediction failed for {eid}: {resp.text}")

    await engine.dispose()
    print("\n🎯 Full loop completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
