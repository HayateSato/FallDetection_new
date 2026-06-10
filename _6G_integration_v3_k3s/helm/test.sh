#!/usr/bin/env bash
# test.sh -- Smoke-test the caregiver deployment after install.
# Run from _6G_integration_v3_k3s/ as working directory.
# Requires: kubectl, curl

set -uo pipefail

NAMESPACE="fall-dashboard"
PASSED=0
FAILED=0

pass() { echo "  PASS  $1"; ((PASSED++)); }
fail() { echo "  FAIL  $1"; ((FAILED++)); }

echo "=== Caregiver smoke tests (namespace: ${NAMESPACE}) ==="
echo ""

# 1. Both pods running
STATUS=$(kubectl get pod -n "${NAMESPACE}" -l app=mosquitto \
    -o jsonpath="{.items[0].status.phase}" 2>/dev/null || true)
[ "${STATUS}" = "Running" ] && pass "mosquitto pod Running" || fail "mosquitto pod Running"

STATUS=$(kubectl get pod -n "${NAMESPACE}" -l app=fall-dashboard \
    -o jsonpath="{.items[0].status.phase}" 2>/dev/null || true)
[ "${STATUS}" = "Running" ] && pass "fall-dashboard pod Running" || fail "fall-dashboard pod Running"

# 2. Services exist
kubectl get svc mosquitto -n "${NAMESPACE}" &>/dev/null \
    && pass "mosquitto Service exists" || fail "mosquitto Service exists"

kubectl get svc fall-dashboard -n "${NAMESPACE}" &>/dev/null \
    && pass "fall-dashboard Service exists" || fail "fall-dashboard Service exists"

# 3. fall-dashboard /api/patients responds (via kubectl port-forward)
echo ""
echo "  Starting port-forward to fall-dashboard:8002 ..."
kubectl port-forward -n "${NAMESPACE}" svc/fall-dashboard 18002:8002 &>/dev/null &
PF_PID=$!
sleep 4

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:18002/api/patients 2>/dev/null || true)
kill "${PF_PID}" 2>/dev/null || true
wait "${PF_PID}" 2>/dev/null || true

[ "${HTTP_CODE}" = "200" ] && pass "GET /api/patients returns 200" || fail "GET /api/patients returns 200 (got: ${HTTP_CODE})"

# 4. IngressRoutes exist
kubectl get ingressroute fall-dashboard -n "${NAMESPACE}" &>/dev/null \
    && pass "fall-dashboard IngressRoute exists" || fail "fall-dashboard IngressRoute exists"

kubectl get ingressroute mosquitto-ws -n "${NAMESPACE}" &>/dev/null \
    && pass "mosquitto WebSocket IngressRoute exists" || fail "mosquitto WebSocket IngressRoute exists"

echo ""
echo "=== Results: ${PASSED} passed, ${FAILED} failed ==="
[ "${FAILED}" -eq 0 ] || exit 1
