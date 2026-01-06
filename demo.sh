#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8000"

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}       PHOENIX ML PLATFORM - DEMO SCENARIO           ${NC}"
echo -e "${BLUE}=====================================================${NC}"

# 1. Health Check
echo -e "\n${GREEN}[1/4] Checking System Health...${NC}"
HEALTH=$(curl -s $BASE_URL/health)
echo "Response: $HEALTH"

# 2. Prediction with Feature Store
echo -e "\n${GREEN}[2/4] Testing Prediction (Enriching features from Store)...${NC}"
# User 'user-123' is seeded in InMemoryStore/Redis on startup
PAYLOAD='{"model_id": "demo-model", "model_version": "v1", "entity_id": "user-123"}'
echo "Payload: $PAYLOAD"
RESPONSE=$(curl -s -X POST "$BASE_URL/predict" \
     -H "Content-Type: application/json" \
     -d "$PAYLOAD")
echo "Response: $RESPONSE"

# 3. Simulate Traffic
echo -e "\n${GREEN}[3/4] Simulating Traffic (50 requests) to generate logs...${NC}"
for i in {1..50}; do
    # Generate random feature values using python inline
    F1=$(python3 -c "import random; print(random.gauss(0, 1))") # Normal distribution
    PAYLOAD_TRAFFIC="{\"model_id\": \"demo-model\", \"model_version\": \"v1\", \"features\": [$F1, 0.5, 0.5, 0.5]}"
    
    curl -s -X POST "$BASE_URL/predict" \
         -H "Content-Type: application/json" \
         -d "$PAYLOAD_TRAFFIC" > /dev/null
    echo -n "."
done
echo " Done!"

# 4. Check Drift
echo -e "\n${GREEN}[4/4] Analyzing Data Drift...${NC}"
# We simulated normal distribution (Mean=0), and reference is also Mean=0
# So Drift should be FALSE
DRIFT_REPORT=$(curl -s "$BASE_URL/monitoring/drift/demo-model")
echo "Drift Report:"
echo $DRIFT_REPORT | python3 -m json.tool

echo -e "\n${BLUE}=====================================================${NC}"
echo -e "${BLUE}                 DEMO COMPLETED                      ${NC}"
echo -e "${BLUE}=====================================================${NC}"
