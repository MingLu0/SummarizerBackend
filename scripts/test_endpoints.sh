#!/usr/bin/env bash
set -euo pipefail

SPACE_URL="${1:?Usage: ./scripts/test_endpoints.sh https://<space>.hf.space}"

echo "==> Testing $SPACE_URL/health"
curl -i "$SPACE_URL/health" || true

echo "==> Testing $SPACE_URL/docs"
curl -i "$SPACE_URL/docs" || true

echo "==> Testing $SPACE_URL/api/v2/summarize/stream"
curl -i -X POST "$SPACE_URL/api/v2/summarize/stream" \
  -H "content-type: application/json" \
  -d '{"text":"This is a test to verify the V2 endpoint responds externally.","max_tokens":50}' || true
