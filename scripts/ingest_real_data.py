import asyncio
import os
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.domain.inference.entities.model import Model
from src.infrastructure.persistence.postgres_model_registry import (
    PostgresModelRegistry,
)
from src.shared.ingestion.data_collector import ExampleCreditDataCollector
from src.shared.ingestion.redis_ingestor import RedisDataIngestor
from src.shared.ingestion.service import IngestionService

SUCCESS_STATUS = 200
DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "credit-risk")
API_URL = os.environ.get("API_URL", "http://localhost:8000")


async def main() -> None:
    print("🚀 Starting REAL-WORLD data pipeline...")

    # 1. Collect Real Data (The 'Crawl' phase)
    collector = ExampleCreditDataCollector()
    real_df = await collector.collect()
    print(f"📊 Collected {len(real_df)} real-world records.")

    # 2. Setup Ingestion (Redis)
    # Using localhost:6380 as mapped in compose.yaml
    ingestor = RedisDataIngestor(redis_url="redis://localhost:6380")
    service = IngestionService(ingestor)

    # Ingest first 50 records as 'real' production features
    features = ["user_age", "account_balance", "credit_score", "purchase_amount"]
    df_cols = ["age", "income", "credit_history", "debt"]
    
    ingest_data = []
    for i in range(50):
        row = real_df.iloc[i]
        ingest_data.append(
            {
                "entity_id": f"real-cust-{i}",
                "data": {f: float(row[df_cols[idx]]) for idx, f in enumerate(features)},
            }
        )

    await service.process_batch(ingest_data)
    print("✅ Ingested 50 real-world feature records into Redis.")

    # 3. Register Model in Docker Postgres (Port 5433)
    _model_id = DEFAULT_MODEL_ID
    _fs_model_id = _model_id.replace("-", "_")
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5433/phoenix"
    )
    engine = create_async_engine(db_url)
    session_factory = async_sessionmaker(bind=engine)

    async with session_factory() as session:
        registry = PostgresModelRegistry(session)
        model_uri = f"local:///models/{_fs_model_id}/v1/model.onnx"
        model_obj = Model(
            id=_model_id,
            version="v1",
            uri=model_uri,
            framework="onnx",
            metadata={"role": "champion", "metrics": {"accuracy": 0.85}},
            created_at=datetime.now(UTC),
            is_active=True,
        )
        await registry.save(model_obj)
        await registry.update_stage(_model_id, "v1", "champion")
        print(f"✅ Registered model '{_model_id}:v1' as Champion in Postgres.")

    # 4. Predict and Verify
    print("\n🔮 Performing predictions on real-world data...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(5):
            eid = f"real-cust-{i}"
            resp = await client.post(
                f"{API_URL}/predict",
                json={"model_id": _model_id, "entity_id": eid},
            )
            if resp.status_code == SUCCESS_STATUS:
                print(f"✔️ Prediction for {eid}: {resp.json()['result']}")
            else:
                print(f"❌ Prediction failed for {eid}: {resp.text}")

    await engine.dispose()
    print("\n🎯 REAL-WORLD pipeline verification complete!")


if __name__ == "__main__":
    asyncio.run(main())
