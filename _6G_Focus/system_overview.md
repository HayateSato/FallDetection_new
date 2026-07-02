# Fall Detection — System Overview

```
                     +------------+   BLE    +--------------------+
    Patient          |  SmarKo    | ──────►  |    Mobile App      |
     Level           |  Wearable  |          |   (React Native)   |
                     +------------+          +----------+---------+
                                                        |
                                    +───────────────────+──────────────────────+
                                     |                   |                      |
                               HTTPS POST           MQTT WSS             InfluxDB write
                               /predict            fall/alert            (biosignals +
                               (run model)         (alerts)               fall events)
                                    |                   |                      |
                                    v                   v                      v
                     +──────────────────────+  +─────────────────────────────────────────────+
                     |      MCS server      |  |              FOCUS K3s cluster              |
                     |                      |  |                                             |
                     |  [ Inference Server ]|  |  [ mosquitto ]          [ FOCUS InfluxDB ]  |
FOCUS Hardware       |    fall detection    |  |    MQTT broker  ──────►   existing infra    |
 /Cloud Level        |    & model API       |  |         |                       |           |
                     |                      |  |         v                       |           |
                     |  (MCS manages this,  |  |  [ fall-dashboard ]  <───────── +           |
                     |   FOCUS does not     |  |    caregiver UI                             |
                     |   need to touch it)  |  |    fall alerts + history                    |
                     +──────────────────────+  +─────────────────────────────────────────────+
```

## What will be added to FOCUS Stack

FOCUS is only responsible for the two services inside the **FOCUS K3s cluster** box:

| Service | What it does |
|---------|-------------|
| `mosquitto` | Receives fall alerts from the mobile app via MQTT over WSS (port 443) |
| `fall-dashboard` | Shows live alerts and fall history to caregivers in a browser |



## Network boundaries

| Connection | Protocol | Who initiates |
|-----------|----------|---------------|
| Wearable → Mobile app | BLE | Wearable |
| Mobile app → MCS inference | HTTPS (port 443) | Mobile app |
| Mobile app → FOCUS mosquitto | MQTT over WSS (port 443) | Mobile app |
| Mobile app → FOCUS InfluxDB | HTTPS (port 443) | Mobile app |
| mosquitto → fall-dashboard | MQTT TCP (port 1883, cluster-internal) | fall-dashboard subscribes |
| fall-dashboard → FOCUS InfluxDB | HTTPS (port 443) | fall-dashboard |

FOCUS only exposes ports **80** and **443** — all traffic goes through Traefik.
