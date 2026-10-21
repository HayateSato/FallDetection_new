# Web App Integration Handover — FOCUS Patient Dashboard

**Audience:** FOCUS DevOps team responsible for the existing Flutter-web Patient Dashboard. The team adds a fall panel to the dashboard. (NOT Isa — Isa is on the mobile-app side; see `04_mobile_app_integration.md`.)
**Repo / branch:** `_6G_Integration_v2_mqtt/` on branch `6G-integration_with_MQTT`.
**Scope:**
- Data contract (REST + SSE)
- What the web app should do
- What is missing (gaps in the web app's integration)
- Where existing pieces live + how to talk to each

This doc covers **only the fall integration**. Demographics, biosignals, and the rest of the Patient Dashboard stay as they are.

---

## 1. Where the Patient Dashboard sits in the system

```
[SmarKo wearable]
      │ BLE
      ▼
[Mobile app on patient's phone]                              ← built by SmarKo / Isa
      │ HTTP POST /predict
      ▼
[inference-server]  in our namespace, port 8001              ← built by us (MCS)
      │ HTTP response: fall_detected=True, observation_id, confidence
      ▼
[Mobile app shows confirmation popup]
      │  patient answers Yes/No + needs_help
      ▼
[Mobile app PUBLISH fall/alert/<patient_id> to MQTT broker]
      │
      ▼
[fall-dashboard]  in our namespace, port 8002                ← built by us (MCS)
      │
      ├── writes one row to Postgres `fall_history` (durable record)
      │
      └── conditionally fans out via SSE to caregiver browsers ──────────►
                                                                          ▼
                                                              [YOUR Patient Dashboard]
                                                              (Flutter web app)
```

**Your only integration points are with our `fall-dashboard` service** — REST (for history) and SSE (for live alerts). You do NOT talk to MQTT, the mobile app, the inference server, or our Postgres directly.

---

## 2. Data contract

### 2.1 What you'll see in payloads

This is the MQTT payload from the mobile app. You don't read it directly, but the same fields surface in our REST/SSE responses, so this is the canonical schema.

```json
{
  "observation_id":    "3a0a603e-cc5a-4355-ad8c-f53f2e4de1b9",
  "patient_id":        "charite-patient-001",
  "device_id":         "6c:1d:eb:04:a9:e6",
  "timestamp":         "2026-04-28T10:00:00+00:00",
  "alert_time":        "2026-04-28T10:00:15+00:00",
  "fall_detected":     true,
  "confidence":        0.8234,
  "model_version":     "v3",
  "patient_confirmed": "yes",
  "needs_help":        true
}
```

| Field | Meaning | Why you care |
|-------|---------|--------------|
| `patient_id` | FHIR Patient identifier | match against patients you already display |
| `timestamp` | when the model detected the fall | display as event time |
| `alert_time` | when the popup completed | display as "patient responded at …" |
| `confidence` | model confidence 0–1 | optional UI element ("85% confidence") |
| `patient_confirmed` | `"yes"` / `"no"` / `"not_answered"` | drives the visual flag state |
| `needs_help` | `true` / `false` / `null` | triggers urgent vs informational styling |
| `observation_id` | UUID linking back to inference | not visually useful; keep for support tickets |

#### The four states and what they mean

| `patient_confirmed` | `needs_help` | What happened | UI treatment |
|---------------------|--------------|---------------|--------------|
| `"yes"` | `true` | confirmed fall + asked for help | **red — urgent action** |
| `"not_answered"` | `null` | popup timed out — patient may be hurt | **amber — possibly serious** |
| `"yes"` | `false` | confirmed fall but patient is okay | informational only — no flag in production UI |
| `"no"` | `null` | false positive — patient said they didn't fall | informational only — no flag |

> **Important:** our SSE stream only delivers the **first two** cases (the ones that need caregiver action). The other two are stored in DB for retraining but do NOT come through SSE. So when you receive an SSE event, it is by definition something a caregiver should look at.

The other two cases do come back via REST `/api/falls?only_falls=false` if you want to surface them in a "history view" — but they don't need a flag.

---

### 2.2 GET /api/falls — fall history

```http
GET https://<our-host>:8002/api/falls?patient_id=charite-patient-001&only_falls=true&limit=200
```

```json
{
  "falls": [
    {
      "id":                42,
      "observation_id":    "3a0a603e-cc5a-4355-ad8c-f53f2e4de1b9",
      "patient_id":        "charite-patient-001",
      "mac_id":            "6c:1d:eb:04:a9:e6",
      "fall_detected":     true,
      "patient_confirmed": "yes",
      "needs_help":        true,
      "detection_time":    "2026-04-28T10:00:00+00:00"
    }
  ]
}
```

| Query param | Default | Notes |
|-------------|---------|-------|
| `patient_id` | none | filter to one patient |
| `only_falls` | `true` | set `false` to include non-fall predictions (rarely useful for the patient dashboard) |
| `limit` | 200 | max 2000 |

Use this on dashboard load to set the recent-fall flag for each patient.

### 2.3 GET /api/stream — live alerts (SSE)

```http
GET https://<our-host>:8002/api/stream
```

A long-lived SSE connection. Each event is a `data: <json>` line — same JSON shape as `/api/falls` rows. Keepalive is sent every 15s as `: keepalive` (an SSE comment, ignore it).

**Filter rule:** SSE only emits the two actionable states (`yes/needs_help=true` and `not_answered`). Everything else is silently dropped server-side. So every SSE event you receive corresponds to a flag you should set immediately.

The first frame after connection is a `event: connected` marker with empty data — ignore it.

### 2.4 CORS

Our service sets `Access-Control-Allow-Origin: *` and accepts `GET / POST / OPTIONS`. The Flutter web app can call us from any origin without proxy. In production we plan to tighten to your specific origin — let us know your dashboard hostname and we'll switch the wildcard.

### 2.5 TLS

The fall_dashboard pod itself serves plain HTTP. TLS termination happens at the FOCUS ingress (Traefik with cert-manager, typically). The Flutter app must use `https://` URLs in production — your ingress unwraps to plain HTTP for the in-cluster hop.

---

## 3. What should happen — responsibility list

The Patient Dashboard must:

| # | Responsibility | Frequency |
|---|----------------|-----------|
| 1 | On dashboard load, fetch `/api/falls?limit=2000` (or filter by current patient list) | once per page load + on user-triggered refresh |
| 2 | For each patient with a recent (last 24h) actionable fall, set the patient card's flag colour | per page load |
| 3 | Subscribe to `/api/stream` (SSE) for live updates | continuous while dashboard is open |
| 4 | When an SSE event arrives, update the relevant patient's flag without a page reload | per event |
| 5 | Visual states: red flag for `yes/needs_help=true`, amber for `not_answered`, no flag for the other two | per event |
| 6 | Reconnect on SSE disconnect with exponential backoff | as needed |
| 7 | (Optional) "View fall history" detail view — fetch `/api/falls?patient_id=<pid>` | per user click |
| 8 | (Optional) Acknowledge / dismiss action on a flag | per user action |
| 9 | (Production) Send `Authorization: Bearer <jwt>` once SSO is wired (TBD) | every request |

Your responsibility ends at "show the flag in the right colour at the right time". Caregiver workflow beyond that (calling the patient, dispatching, escalating) is human/process — see `07_user_flow_caregiver.md`.

---

## 4. What is missing (gaps to close)

What the Patient Dashboard does NOT yet implement on the fall side (as of 2026-04-29):

| Gap | Status | Notes |
|-----|:------:|-------|
| `flutter_client_sse` package added to `pubspec.yaml` | not started | Or whichever SSE client your stack prefers. |
| Initial REST fetch of `/api/falls` on dashboard load | not started | See Section 5.2 for Dart code. |
| SSE subscription to `/api/stream` | not started | See Section 5.3 for Dart code. |
| Patient card flag rendering (red / amber / none) | not started | The two helpers `_isActionable()` + `_flagColor()` encode it once. |
| Auto-reconnect on SSE disconnect | partially — `flutter_client_sse` handles this for free | Confirm in your environment. |
| Time-based flag expiry (24h cutoff) | not started | We recommend you do this client-side; the server doesn't track it. |
| "View fall history" detail screen | not started | Consume `/api/falls?patient_id=<pid>` for the table. |
| (Optional) "Mark resolved" / "Acknowledge" action | not started | The server does not currently track an "acknowledged" state. If you want this we can add an endpoint — flag for discussion. |
| JWT-based auth header on requests | not started | Blocks on FOCUS SSO format (see Section 6). Currently no auth — gate at ingress. |
| Production ingress URL configured | not started | Blocks on FOCUS DevOps assigning the public hostname. |

What is **already implemented** on the fall_dashboard backend and you don't need to ask us for:

- REST `/api/falls` (with `patient_id`, `only_falls`, `limit` query params)
- SSE `/api/stream` (with server-side filter, keepalive, reconnect-friendly)
- CORS wildcard (will tighten in production)
- Postgres `fall_history` storage of all four cases (so historical view is complete)

---

## 5. How to integrate — Flutter / Dart code

### 5.1 Add packages

In `pubspec.yaml`:

```yaml
dependencies:
  http: ^1.2.0                  # REST client
  flutter_client_sse: ^2.0.0    # SSE client (works on Flutter web)
```

`flutter_client_sse` is the most-used SSE package for Flutter web at the time of writing. If you already use a different HTTP/streaming stack, the patterns below port directly — what matters is the data flow.

### 5.2 REST — load fall history on dashboard load

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class Fall {
  final String  patientId;
  final String? macId;
  final String  patientConfirmed;     // "yes" | "no" | "not_answered"
  final bool?   needsHelp;
  final DateTime detectionTime;
  final String  observationId;

  Fall.fromJson(Map<String, dynamic> j)
      : patientId        = j['patient_id'],
        macId            = j['mac_id'],
        patientConfirmed = j['patient_confirmed'],
        needsHelp        = j['needs_help'],
        detectionTime    = DateTime.parse(j['detection_time']),
        observationId    = j['observation_id'];
}

Future<List<Fall>> fetchRecentFalls() async {
  final uri = Uri.parse('https://<your-fall-dashboard-host>/api/falls?limit=2000');
  final r   = await http.get(uri);
  if (r.statusCode != 200) {
    throw Exception('GET /api/falls failed: ${r.statusCode}');
  }
  final body  = jsonDecode(r.body) as Map<String, dynamic>;
  final falls = (body['falls'] as List).cast<Map<String, dynamic>>();
  return falls.map(Fall.fromJson).toList();
}
```

### 5.3 SSE — subscribe to live updates

```dart
import 'package:flutter_client_sse/flutter_client_sse.dart';

void connectFallStream(void Function(Fall) onFall) {
  SSEClient.subscribeToSSE(
    method: SSERequestType.GET,
    url:    'https://<your-fall-dashboard-host>/api/stream',
    header: {
      'Accept':        'text/event-stream',
      'Cache-Control': 'no-cache',
    },
  ).listen((event) {
    if (event.data == null || event.data!.trim().isEmpty) return;
    try {
      final j    = jsonDecode(event.data!) as Map<String, dynamic>;
      final fall = Fall.fromJson(j);
      onFall(fall);
    } catch (e) {
      print('Bad SSE payload: ${event.data}');
    }
  });
}
```

### 5.4 Wiring it into the dashboard widget

```dart
class PatientDashboardScreen extends StatefulWidget {
  @override
  _PatientDashboardScreenState createState() => _PatientDashboardScreenState();
}

class _PatientDashboardScreenState extends State<PatientDashboardScreen> {
  // patient_id → most recent unresolved fall
  final Map<String, Fall> _flags = {};

  @override
  void initState() {
    super.initState();
    _loadInitialState();
    connectFallStream(_handleNewFall);
  }

  Future<void> _loadInitialState() async {
    final falls  = await fetchRecentFalls();
    final cutoff = DateTime.now().subtract(const Duration(hours: 24));

    setState(() {
      for (final f in falls) {
        if (f.detectionTime.isBefore(cutoff)) continue;
        if (_isActionable(f)) _flags[f.patientId] = f;
      }
    });
  }

  void _handleNewFall(Fall fall) {
    if (!_isActionable(fall)) return;     // SSE filters server-side; defence in depth
    setState(() => _flags[fall.patientId] = fall);
  }

  bool _isActionable(Fall f) {
    return f.patientConfirmed == 'not_answered' ||
           (f.patientConfirmed == 'yes' && f.needsHelp == true);
  }

  Color _flagColor(Fall? f) {
    if (f == null)                                                 return Colors.transparent;
    if (f.patientConfirmed == 'yes' && f.needsHelp == true)       return Colors.red;
    if (f.patientConfirmed == 'not_answered')                      return Colors.amber;
    return Colors.transparent;
  }

  // build() renders patient cards and applies _flagColor() to each
}
```

The two helpers `_isActionable()` + `_flagColor()` encode the rules once. Replace the colours with whatever your design system uses.

---

## 6. Where existing pieces live + how to talk to each

### 6.1 What you call (and don't call)

| You call | Component | Protocol | Where in production |
|----------|-----------|----------|---------------------|
| `GET /api/falls`, `GET /api/stream` | **fall-dashboard (ours)** | HTTPS / SSE | `https://<focus-ingress-host>/api/falls`, `https://<focus-ingress-host>/api/stream` (path TBD by FOCUS) |
| FHIR Patient demographics | **FOCUS FHIR server** | HTTPS | existing — unchanged |
| Biosignals (HR etc.) | **FOCUS InfluxDB** | InfluxDB HTTP API | existing — unchanged |
| (Optional) FHIR Observations for falls | **FOCUS FHIR server** | HTTPS | optional — only if MCS is configured to push (not your concern) |

| You do NOT call | Why |
|-----------------|-----|
| MQTT broker | The mobile app is publish-only; you consume the resulting REST/SSE |
| inference-server | That's the mobile app's concern |
| MLflow / MinIO / Postgres | Internal-only for MCS |

### 6.2 The cross-namespace URL question

The Flutter web app runs in the **user's browser**, not in the cluster. So in-cluster service DNS (`fall-dashboard.mcs-fall-detection.svc.cluster.local:8002`) is irrelevant — that's only for pod-to-pod traffic.

Your Flutter app calls our service via the **public ingress hostname** that FOCUS DevOps assigns:

| Environment | URL |
|-------------|-----|
| Local dev | `http://localhost:8002` |
| FOCUS production | `https://<your-ingress-host>/api/falls/...` (path-routed by Traefik to our `fall-dashboard` Service) |

The exact production URL is something **you decide** — you control the ingress. Tell us which path you want to route to our service (e.g. `/api/falls/*` → `fall-dashboard.mcs-fall-detection.svc.cluster.local:8002`) and we'll align our chart's expected base path if it differs.

If you'd rather give us a dedicated subdomain (`fall-api.charite.de`), that works too — same Service, just a different ingress rule.

### 6.3 Reference implementations

| Where to look | What it shows |
|---------------|---------------|
| `_6G_Integration_v2_mqtt/fall_dashboard/web.py` | The actual server-side endpoints (`/api/falls`, `/api/stream`) — you can read the response shape directly |
| `_6G_Integration_v2_mqtt/fall_dashboard/dashboard/app.js` | A vanilla-JS reference frontend that consumes the same APIs you will |
| `_6G_Integration_v2_mqtt/helm/mock-focus/charts/mock-patient-dashboard/` | A FastAPI proxy + HTML SPA that runs in a fake "FOCUS namespace" and consumes our SSE feed cross-namespace — proves the wiring works |

If you fork the JS frontend, the `app.js` file is ~300 lines and shows the same flow this doc describes.

---

## 7. Local dev — testing before production

Run our stack locally and point your Flutter dev build at it:

```powershell
# In a terminal, from _6G_Integration_v2_mqtt/ as cwd

# 1. Start infrastructure (Postgres, MQTT, etc.)
docker-compose -f infrastructure/docker-compose.yml up

# 2. Start inference-server
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001

# 3. Start fall-dashboard
python -m fall_dashboard.main      # http://localhost:8002

# 4. Start mock_app (publishes simulated falls on a schedule)
python -m local_dev.mock_app.main  # popup at http://localhost:8005
```

Then in your Flutter dev:

```powershell
flutter run -d chrome --web-port 5000
```

Configure your Dart code to call `http://localhost:8002` for fall data. To trigger a fall:

1. Open `http://localhost:8005/` (the mock popup).
2. Click "Yes I fell" → "Yes I need help".
3. Within 1–2 seconds your Flutter app's SSE stream fires; the corresponding patient card flips to red.

If that round-trip works locally, the integration is functionally complete. Production deployment then becomes ingress + URL config only.

---

## 8. Auth — what's there now vs planned

**Today (dev):** `/api/falls` and `/api/stream` have **no auth**. CORS is wildcard. Anyone who can reach the endpoints can read all fall data.

**Production plan (todo.md Step 11.5.4):**
- FOCUS SSO issues a JWT to the user.
- Your dashboard sends the JWT as `Authorization: Bearer <token>` on every request.
- Our `fall_dashboard` validates the JWT — checks `iss` and the `role` claim.
- Caregivers see fall data; admins do not.

We need from FOCUS: JWT format your SSO uses (`role` claim name, signing algorithm, JWKS URL). Once that's settled, the auth gate is a small change on our side.

Until then: gate access at the **ingress level** (Traefik basic auth, IP allow-list, mTLS — whatever fits your cluster's existing security posture). Don't expose our endpoints to the open internet without something in front.

---

## 9. End-to-end checklist

When the integration is finished, all of these should be true:

- [ ] On dashboard load, each patient with a recent (last 24h) actionable fall shows a coloured flag
- [ ] When a fall fires live (mobile app → MQTT → our SSE), the patient flag updates within 1–2s without a page reload
- [ ] Red flag = `patient_confirmed='yes' AND needs_help=true` (urgent)
- [ ] Amber flag = `patient_confirmed='not_answered'` (patient unresponsive)
- [ ] False-positive falls (`patient_confirmed='no'`) do NOT flag (and do NOT come through SSE — verify by triggering one)
- [ ] "Okay" confirmations (`patient_confirmed='yes' AND needs_help=false`) do NOT flag
- [ ] If the dashboard reloads, the flag persists (because the REST `/api/falls` query rebuilds it from history — not relying on SSE alone)
- [ ] CORS works without a proxy
- [ ] Production traffic uses HTTPS + ingress with auth (whatever you wire up)

---

## 10. Common pitfalls (read once)

- **Time format:** `detection_time` is ISO 8601 with timezone. `DateTime.parse()` in Dart handles it. If you see "1970-01-01" in your UI, you're parsing the wrong field.
- **`patient_confirmed` is a string, not an enum.** Compare with `==`, not bitwise.
- **`needs_help` can be `null`** (when `patient_confirmed == "not_answered"`). Treat as a tri-state, not a bool.
- **SSE keepalives:** every 15s the server sends `: keepalive` (a comment, not an event). The `flutter_client_sse` package ignores these automatically. If you write a custom client, treat lines starting with `:` as comments.
- **Reconnect storms:** if our service is down, the SSE client will retry forever. Add an exponential backoff if you've extended the package.
- **DevTools → Network tab:** check `/api/stream` shows `(pending)` with type `eventsource` and a 200 status — that means the connection is alive.
- **Flag persistence on reload:** rely on the REST fetch on load, not on SSE replay. SSE does not deliver historical events.

---

## 11. Cross-references

- [`03_fall_detection_system.md`](03_fall_detection_system.md) — system architecture, where the Patient Dashboard fits
- [`04_mobile_app_integration.md`](04_mobile_app_integration.md) — what the mobile app pushes upstream of you
- [`07_user_flow_caregiver.md`](07_user_flow_caregiver.md) — how the caregiver uses what your dashboard renders
- Existing handover doc at `handover_docs/FOCUS_patient_dashboard_integration.md` — earlier version of this content

---

## 12. Contact

| For | Reach out to |
|-----|--------------|
| API contract questions, schema changes | Hayate (MCS) |
| Mobile app behaviour (what triggers a fall flag) | Isa (SmarKo) — see `04_mobile_app_integration.md` |
| Production hostname / ingress / TLS / auth integration | FOCUS DevOps internal team (you) |
| Patient demographics, biosignals (out of scope) | the existing FHIR + InfluxDB integration in your dashboard — unchanged |

A 30-minute sync once you've started integration usually clears 80% of remaining questions.
