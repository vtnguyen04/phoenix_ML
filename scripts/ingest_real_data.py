import asyncio
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.domain.inference.entities.model import Model
from src.infrastructure.persistence.postgres_model_registry import (
    PostgresModelRegistry,
)
from src.shared.ingestion.data_collector import CreditDataCollector
from src.shared.ingestion.redis_ingestor import RedisDataIngestor
from src.shared.ingestion.service import IngestionService

SUCCESS_STATUS = 200


async def main() -> None:
    print("🚀 Starting REAL-WORLD data pipeline...")

    # 1. Collect Real Data (The 'Crawl' phase)
    collector = CreditDataCollector()
    real_df = await collector.collect()
    print(f"📊 Collected {len(real_df)} real-world records.")

    # 2. Setup Ingestion (Redis)
    # Using localhost:6380 as mapped in compose.yaml
    ingestor = RedisDataIngestor(redis_url="redis://localhost:6380")
    service = IngestionService(ingestor)

    # Ingest first 50 records as 'real' production features
    ingest_data = []
    for i in range(50):
        row = real_df.iloc[i]
        ingest_data.append(
            {
                "entity_id": f"real-cust-{i}",
                "data": {f"f{j+1}": float(row[f"f{j+1}"]) for j in range(4)},
            }
        )

    await service.process_batch(ingest_data)
    print("✅ Ingested 50 real-world feature records into Redis.")

    # 3. Register Model in Docker Postgres (Port 5433)
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
            metadata={"role": "champion", "metrics": {"accuracy": 0.85}},
            created_at=datetime.now(UTC),
            is_active=True,
        )
        await registry.save(credit_model)
        await registry.update_stage("credit-risk", "v1", "champion")
        print("✅ Registered model 'credit-risk:v1' as Champion in Postgres.")

    # 4. Predict and Verify
    print("\n🔮 Performing predictions on real-world data...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(5):
            eid = f"real-cust-{i}"
            resp = await client.post(
                "http://localhost:8001/predict",
                json={"model_id": "credit-risk", "entity_id": eid},
            )
            if resp.status_code == SUCCESS_STATUS:
                print(f"✔️ Prediction for {eid}: {resp.json()['result']}")
            else:
                print(f"❌ Prediction failed for {eid}: {resp.text}")

    await engine.dispose()
    print("\n🎯 REAL-WORLD pipeline verification complete!")


if __name__ == "__main__":
    asyncio.run(main())
