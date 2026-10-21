# Patient Dashboard — Fall Integration Guide for FOCUS DevOps

**Audience:** the FOCUS DevOps team responsible for the existing Flutter-web Patient Dashboard.
**Scope:** how to add fall-detection data (history, live alerts, status flags) to the Patient Dashboard. This is the **only** thing this doc covers — demographics, biosignals, and the rest of your dashboard stay as they are.
**Not for:** Isa (he's on the mobile-app side now — see [`ISA_mobile_app_contract.md`](ISA_mobile_app_contract.md)). Not for our internal admin tools (those are separate dashboards we own — `ml_dashboard` and `server_health`).

---

## 1. The end-to-end flow you're plugging into

```
[SmarKo wearable]
      │ BLE
      ▼
[Mobile app on patient's phone]                       ← built by SmarKo / Isa
      │ HTTP POST /predict
      ▼
[inference_server]  in our namespace, port 8001       ← built by us (MCS)
      │ HTTP response: fall_detected=True, observation_id, confidence
      ▼
[Mobile app shows confirmation popup]
      │  patient answers Yes/No + needs_help
      ▼
[Mobile app PUBLISH fall/alert/<patient_id> to MQTT broker]
      │
      ▼
[fall_dashboard]  in our namespace, port 8002         ← built by us (MCS)
      │
      ├── writes one row to Postgres `fall_history` (durable record)
      │
      └── conditionally fans out via SSE to caregiver browsers ──────────►
                                                                          ▼
                                                              [YOUR Patient Dashboard]
                                                              (Flutter web app)
```

You sit at the bottom right. The fall-detection pipeline is upstream of you — by the time the data reaches your dashboard it's already been (a) inferred by the model, (b) confirmed by the patient, and (c) persisted in our Postgres.

**Your only integration points are with our `fall_dashboard` service** — REST (for history) and SSE (for live alerts). You do not talk to MQTT, the mobile app, the inference server, or our database directly.

---

## 2. What data the mobile app passes when the patient confirms

This is the MQTT payload published by the mobile app after the popup closes.
You don't consume MQTT directly, but it's useful to know what fields exist
because the same fields appear in our REST/SSE responses.

```json
{
  "observation_id":    "3a0a603e-cc5a-4355-ad8c-f53f2e4de1b9",
  "patient_id":        "charite-patient-001",
  "device_id":         "6c:1d:eb:04:a9:e6",
  "timestamp":         "2026-04-28T10:00:00+00:00",
  "alert_time":        "2026-04-28T10:00:15+00:00",
  "fall_detected":     true,
  "confidence":        0.8234,
  "model_version":     "v0",
  "patient_confirmed": "yes",
  "needs_help":        true
}
```

Field meanings:

| Field | Meaning | Why you care |
|-------|---------|--------------|
| `patient_id` | FHIR Patient identifier | match this against patients you already display |
| `timestamp` | when the model detected the fall | display as event time |
| `alert_time` | when the popup completed | display as "patient responded at …" |
| `confidence` | model confidence score 0–1 | optional UI element ("85% confidence") |
| `patient_confirmed` | `"yes"` / `"no"` / `"not_answered"` | the visual state of the flag |
| `needs_help` | `true` / `false` / `null` | triggers urgent vs informational styling |
| `observation_id` | UUID linking back to the inference | not visually useful; keep for support tickets |

The four possible state combinations:

| `patient_confirmed` | `needs_help` | What happened | UI treatment |
|---------------------|--------------|---------------|--------------|
| `"yes"` | `true` | confirmed fall + asked for help | **red — urgent action** |
| `"not_answered"` | `null` | popup timed out — patient may be hurt | **amber — possibly serious** |
| `"yes"` | `false` | confirmed fall but patient is okay | informational only — usually no flag |
| `"no"` | `null` | false positive — patient said they didn't fall | informational only — usually no flag |

**Important:** our SSE stream only delivers the **first two** cases (the ones that need caregiver action). The other two are stored in DB for retraining but do NOT come through SSE. So when you receive an SSE event, it is by definition something a caregiver should look at.

---

## 3. The two endpoints you'll consume

Our `fall_dashboard` exposes a small REST + SSE surface. CORS is wide-open
(`Access-Control-Allow-Origin: *`) so a Flutter web app served from a
different origin can call us without proxy.

### 3a. `GET /api/falls` — fall history

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

Use this on dashboard load to show the recent-fall flag for each patient. Optional query params:

- `patient_id` — filter to one patient
- `only_falls` (default `true`) — set to `false` if you want to see non-fall predictions too (rarely useful for the patient dashboard)
- `limit` (default 200, max 2000)

### 3b. `GET /api/stream` — live alerts (Server-Sent Events)

```http
GET https://<our-host>:8002/api/stream
```

A long-lived SSE connection. Each event contains the same JSON shape as `/api/falls` rows. Reconnect automatically on drop — the browser/Flutter client side handles this for you.

**Filter rule:** SSE only emits events that need caregiver action (the two red/amber states). False positives and "okay" confirmations are silently filtered out on our side. So every event you receive corresponds to a patient flag you should set immediately.

---

## 4. Flutter integration — Dart code

Two pieces: REST fetch on load, SSE for live updates.

### 4a. Add packages

In `pubspec.yaml`:

```yaml
dependencies:
  http: ^1.2.0                  # REST client
  flutter_client_sse: ^2.0.0    # Server-Sent Events client (works on Flutter web)
```

`flutter_client_sse` is the most-used SSE package for Flutter web at the time of writing. If you already use a different HTTP/streaming stack, the patterns below port directly — what matters is the data flow.

### 4b. REST — load fall history

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class Fall {
  final String patientId;
  final String? macId;
  final String patientConfirmed;     // "yes" | "no" | "not_answered"
  final bool? needsHelp;
  final DateTime detectionTime;
  final String observationId;

  Fall.fromJson(Map<String, dynamic> j)
      : patientId        = j['patient_id'],
        macId            = j['mac_id'],
        patientConfirmed = j['patient_confirmed'],
        needsHelp        = j['needs_help'],
        detectionTime    = DateTime.parse(j['detection_time']),
        observationId    = j['observation_id'];
}

Future<List<Fall>> fetchRecentFalls() async {
  final uri = Uri.parse('https://<our-host>:8002/api/falls?limit=2000');
  final r = await http.get(uri);
  if (r.statusCode != 200) {
    throw Exception('GET /api/falls failed: ${r.statusCode}');
  }
  final body = jsonDecode(r.body) as Map<String, dynamic>;
  final falls = (body['falls'] as List).cast<Map<String, dynamic>>();
  return falls.map(Fall.fromJson).toList();
}
```

### 4c. SSE — live updates

```dart
import 'package:flutter_client_sse/flutter_client_sse.dart';

void connectFallStream(void Function(Fall) onFall) {
  SSEClient.subscribeToSSE(
    method: SSERequestType.GET,
    url: 'https://<our-host>:8002/api/stream',
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
      // Bad payload — log and ignore
      print('Bad SSE payload: ${event.data}');
    }
  });
}
```

The package handles automatic reconnect on disconnect. The first event after connection is a marker (`event: connected`, empty data) that you can ignore.

### 4d. Wiring it into your dashboard widget

```dart
class PatientDashboardScreen extends StatefulWidget {
  @override
  _PatientDashboardScreenState createState() => _PatientDashboardScreenState();
}

class _PatientDashboardScreenState extends State<PatientDashboardScreen> {
  // patient_id → most recent unresolved fall (null if no flag)
  final Map<String, Fall> _flags = {};

  @override
  void initState() {
    super.initState();
    _loadInitialState();
    connectFallStream(_handleNewFall);
  }

  Future<void> _loadInitialState() async {
    final falls = await fetchRecentFalls();
    final cutoff = DateTime.now().subtract(const Duration(hours: 24));

    setState(() {
      for (final f in falls) {
        if (f.detectionTime.isBefore(cutoff)) continue;
        if (_isActionable(f)) _flags[f.patientId] = f;
      }
    });
  }

  void _handleNewFall(Fall fall) {
    if (!_isActionable(fall)) return;     // SSE filters this server-side already, but defence in depth
    setState(() => _flags[fall.patientId] = fall);
  }

  bool _isActionable(Fall f) {
    return f.patientConfirmed == 'not_answered' ||
           (f.patientConfirmed == 'yes' && f.needsHelp == true);
  }

  Color _flagColor(Fall? f) {
    if (f == null) return Colors.transparent;
    if (f.patientConfirmed == 'yes' && f.needsHelp == true) return Colors.red;
    if (f.patientConfirmed == 'not_answered')                return Colors.amber;
    return Colors.transparent;
  }

  // ... build() renders a patient list and applies _flagColor() to each card
}
```

The two helpers `_isActionable()` and `_flagColor()` encode the colour rules once, so you don't have to repeat them everywhere. Replace the colours with whatever your design system uses.

---

## 5. Cross-namespace networking — what URL does the Flutter app actually call?

The Flutter web app runs in the **user's browser**, not inside the cluster. So in-cluster service DNS (`fall-dashboard.mcs-fall-detection.svc.cluster.local:8002`) is irrelevant for your case — that's only used by pods talking to other pods.

Your Flutter app calls our service via the **public ingress hostname** that FOCUS DevOps assigns:

| Environment | URL the Flutter app uses |
|-------------|--------------------------|
| Local dev | `http://localhost:8002` (we run on your machine alongside) |
| FOCUS production | `https://<your-ingress-host>/falls/...` (path-routed by Traefik to our `fall-dashboard` Service) |

The exact production URL is something **you decide** — you control the ingress. Tell us which path you want to route to our service (e.g. `/api/falls/*` → `fall-dashboard.mcs-fall-detection.svc.cluster.local:8002`) and we'll align our chart's expected base path if needed.

If you'd rather give us a dedicated subdomain (`fall-api.charite.de`), that works too — same Service, just a different ingress rule on your side.

### CORS

Our service sets `Access-Control-Allow-Origin: *` and accepts GET/POST/OPTIONS. So the browser can call us from any origin without preflight blocking. In production you may want to tighten this to only your dashboard's origin — let us know and we'll switch the wildcard to your hostname.

### TLS

The `fall_dashboard` container itself serves plain HTTP. TLS termination is expected at your ingress (Traefik with cert-manager, typically). The Flutter app must use `https://...` URLs — your ingress unwraps to plain HTTP for the in-cluster hop.

---

## 6. Auth — what's there now vs. planned

**Today (dev):** the `/api/falls` and `/api/stream` endpoints have **no auth**. Anyone who can reach them can read all fall data. CORS is wildcard.

**Production plan (todo.md Step 11.5.4):**
- FOCUS SSO issues a JWT to the user
- Your dashboard sends the JWT as `Authorization: Bearer <token>` on every request
- Our `fall_dashboard` validates the JWT — checks `iss` and the `role` claim
- Caregivers see fall data; admins do not

We'd love your input on the JWT format your SSO uses (`role` claim name, signing algorithm, JWKS URL). Once that's settled, the auth gate is a small change on our side.

Until then: gate access at the **ingress level** (Traefik basic auth, IP allow-list, mTLS — whatever fits your cluster's existing security posture). Don't expose our endpoints to the open internet without something in front.

---

## 7. Local dev — how to test before deploying

Run our stack locally and point your Flutter dev build at it:

1. We start our stack: `docker-compose -f infrastructure/docker-compose.yml up` then a few `python -m ...` commands (covered in our `_6G_Integration_v2_mqtt/README.md`).
2. We run `mock_app` which simulates the SmarKo mobile app — it generates fall events on a schedule.
3. You run your Flutter web app with `flutter run -d chrome --web-port 5000`.
4. Configure your Dart code to call `http://localhost:8002` for fall data.
5. Trigger a fall: open `http://localhost:8005/` (our patient confirmation popup mock). Click "Yes I fell" → "Yes need help".
6. Within 1–2 seconds your Flutter app's SSE stream fires; the corresponding patient card flips to red.

If that round-trip works locally, the integration is functionally complete. Production deployment then becomes ingress + URL config only.

---

## 8. End-to-end checklist

When the integration is finished, all of these should be true:

- [ ] On dashboard load, each patient with a recent (last 24h) actionable fall shows a coloured flag
- [ ] When a fall fires live (mobile app → MQTT → our SSE), the patient flag updates within 1–2s without a page reload
- [ ] Red flag = `patient_confirmed='yes' AND needs_help=true` (urgent action)
- [ ] Amber flag = `patient_confirmed='not_answered'` (patient unresponsive)
- [ ] False-positive falls (`patient_confirmed='no'`) do NOT flag anything (and do NOT come through SSE — verify by triggering one)
- [ ] "Okay" confirmations (`patient_confirmed='yes' AND needs_help=false`) do NOT flag anything either
- [ ] If your dashboard reloads, the flag persists (because the REST `/api/falls` query rebuilds it from history — not relying on SSE alone)
- [ ] CORS works without a proxy
- [ ] Production traffic uses HTTPS + ingress with auth (whatever you wire up)

---

## 9. Common pitfalls (from our experience helping Isa)

- **Time format:** `detection_time` is ISO 8601 with timezone. `DateTime.parse()` in Dart handles it. If you see "1970-01-01" in your UI, you're parsing the wrong field.
- **`patient_confirmed` is a string, not an enum.** Compare with `==`, not bitwise.
- **`needs_help` can be `null`** (when `patient_confirmed == "not_answered"`). Treat it as `null/false/true` tri-state, not just bool.
- **SSE keepalives:** every 15s our server sends `: keepalive` (a comment, not an event). The `flutter_client_sse` package ignores these automatically. If you write a custom client, treat lines starting with `:` as comments.
- **Reconnect storms:** if our service is down, the SSE client will retry forever. Add an exponential backoff if you've extended the package.
- **Browser DevTools tab:** Network → check the `/api/stream` request shows `(pending)` with type `eventsource` and a 200 status — that means the connection is alive.

---

## 10. Contact

| For | Reach out to |
|-----|--------------|
| API contract questions, schema changes | Hayate (MCS) |
| Mobile app behaviour (what triggers an MQTT publish) | Isa (mobile-app side) — see `ISA_mobile_app_contract.md` |
| Production hostname / ingress / TLS / auth integration | FOCUS DevOps internal team (you) |
| Patient demographics, biosignals (out of scope here) | the existing FHIR + InfluxDB integration in your dashboard — unchanged |

Happy to be on a call once you've started integration — typically a 30-min sync clears the last 80% of questions.
