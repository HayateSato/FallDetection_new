# Redis — What It Is and Why It's Used Here

Educational reference for the Fall Detection multi-user system.

---

## What Is Redis?

Redis stands for **RE**mote **DI**ctionary **S**erver. It is an in-memory data store — all data lives in RAM, which is what makes it extremely fast (sub-millisecond reads and writes).

Redis can act as:
- A **key-value store** (like a very fast dictionary)
- A **cache** (store temporary results so you don't recompute them)
- A **message broker** via its **pub/sub** system
- A **queue** (via lists)

In this project, only the **pub/sub** feature is used.

---

## What Is Pub/Sub?

Pub/sub stands for **publish / subscribe**. It is a messaging pattern where:

- A **publisher** sends a message to a named **channel**
- Any number of **subscribers** listening on that channel receive the message immediately
- The publisher and subscriber do **not know about each other** — they only agree on a channel name

```
Publisher                   Redis Channel              Subscribers
─────────                   ─────────────              ───────────
ml_server  ──► publish ──►  "fall_events"  ──► listen ──► caregiver_api
                                           ──► listen ──► emergency_svc
```

Redis acts as the **middleman**. The sender just calls `publish`, and every service that has called `subscribe` on that channel gets the message pushed to them.

---

## How Redis Is Used in This Project

### The channel: `fall_events`

When the ML server (`system_operator/ml_server/server.py`) finishes a prediction and detects a fall, it publishes one JSON message to Redis:

```python
r.publish("fall_events", json.dumps({
    "patient_id": "alice",
    "fall_detected": True,
    "confidence": 0.93,
    "model_version": "v3",
    "timestamp": "2026-03-23T14:32:01+00:00",
    "inference_id": 42,
}))
```

Two services subscribe to this channel and react:

| Subscriber | File | What it does with the event |
|------------|------|-----------------------------|
| Caregiver API | `caregiver/api/server.py` | Pushes the alert to caregiver browser dashboards via SSE |
| Emergency Service | `emergency/notification_service/server.py` | Pushes the alert to all connected tablets via SSE, and optionally to a webhook |

### The flow in full

```
1. Sensor data arrives → ml_server runs XGBoost → fall detected (confidence 0.93)
2. ml_server publishes JSON to Redis channel "fall_events"   ← happens in <1ms
3a. caregiver_api receives message → pushes to caregiver browser → alert banner appears
3b. emergency_svc receives message → pushes to emergency tablet → full-screen alert
```

Steps 3a and 3b happen in parallel, automatically, without ml_server doing anything extra.

---

## Why Not Just Call the Services Directly?

The alternative is a **direct HTTP call**: after each prediction, ml_server would do:

```python
# Without Redis — direct calls
requests.post("http://caregiver_api:8002/internal/fall", json=payload)
requests.post("http://emergency_svc:8003/internal/fall", json=payload)
```

This works but has serious problems in production:

| Problem | With direct HTTP | With Redis pub/sub |
|---------|------------------|--------------------|
| Caregiver API is down | ml_server request fails or hangs | Redis buffers nothing — but ml_server never blocks or errors |
| You add a 3rd subscriber | You must edit ml_server code | New service just subscribes to the channel, no code change anywhere else |
| ml_server inference latency | Waits for both HTTP calls to finish | Redis publish returns instantly; subscribers are independent |
| One slow subscriber | Blocks ml_server | Each subscriber consumes at its own pace |

The key principle is **decoupling**: ml_server is responsible only for making predictions. It does not know or care how many services consume the result, or whether they are up. This makes the system easier to extend and more fault-tolerant.

---

## Why Redis Specifically (vs. Other Brokers)?

More powerful alternatives like **Kafka** or **RabbitMQ** are common in large production systems. Redis was chosen here for several practical reasons:

| | Redis | Kafka | RabbitMQ |
|-|-------|-------|----------|
| Setup | Single Docker container, zero config | Needs ZooKeeper/KRaft cluster | Needs separate broker config |
| Learning curve | Very low | High | Medium |
| Message persistence | No (pub/sub is fire-and-forget) | Yes (log-based, days of history) | Yes (queues with acks) |
| Suitable for this project | Yes — alert latency matters more than durability | Overkill | Overkill |
| Doubles as cache/session store | Yes | No | No |

For this project, **if a fall alert is missed because a subscriber was restarting, that is acceptable** — the event is already written to PostgreSQL and the caregiver can see it in the dashboard history. The real-time notification path is best-effort, and Redis matches that requirement with minimal operational overhead.

---

## Redis in the Docker Compose Stack

The Redis container is defined in `infrastructure/docker-compose.yml`:

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
```

All Python services connect using the `REDIS_URL` environment variable:

```
REDIS_URL=redis://localhost:6379/0
```

The `/0` at the end is the **database index** — Redis supports 16 logical databases (0–15) inside one server. Using `0` is the default.

---

## The Code in This Project

### Publishing (ml_server)

Located in `system_operator/ml_server/server.py`, inside `_publish_fall_event()`:

```python
r = redis.from_url(REDIS_URL)
r.publish("fall_events", payload_json)
```

This runs inside a `BackgroundTasks` call, so it happens **after** the HTTP response is sent to the client. The inference latency is never affected.

### Subscribing (shared helper)

Located in `shared/redis_client.py`, the `subscribe_fall_events()` async generator:

```python
async for event in subscribe_fall_events():
    # event is already a decoded Python dict
    await manager.broadcast(event)   # push to all SSE clients
```

### SSE fan-out (emergency service)

Located in `emergency/notification_service/channels/sse.py`. Each browser tab that opens `/stream` gets its own `asyncio.Queue`. When Redis delivers a message, `manager.broadcast()` puts it into every connected queue simultaneously. Each SSE response handler drains its own queue and sends `data:` lines to the browser.

---

## Summary

Redis solves one specific problem in this system: **how does a single ML prediction reach multiple independent services in real time, without coupling them together?**

The answer: publish to a named channel. Every interested service subscribes. Redis delivers the message in under a millisecond. No service knows about any other service — they only share the channel name `"fall_events"` and the agreed JSON schema.

This pattern (pub/sub via an in-memory broker) is standard in event-driven microservice architectures. Learning it here on a small system gives you a direct mental model for how the same pattern works at scale in systems like Kafka.
