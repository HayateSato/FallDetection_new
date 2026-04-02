# Patient — Reference Documentation

This folder contains no executable code. It documents the patient-side hardware and app setup.

## Hardware

| Component | Details |
|-----------|---------|
| Wearable | SmarKo IMU watch |
| Sensors | Bosch accelerometer at 25 Hz, barometer at 25 Hz |
| Connectivity | Bluetooth Low Energy to smartphone |

## Smartphone App (SmarKo)

The SmarKo app (iOS / Android) receives sensor data from the wearable over BT and uploads it to InfluxDB.

**InfluxDB fields written by the app:**

| Field | Unit | Notes |
|-------|------|-------|
| `bosch_acc_x` | raw LSB (int) | Bosch sensitivity: ±8g = 16,384 LSB |
| `bosch_acc_y` | raw LSB (int) | |
| `bosch_acc_z` | raw LSB (int) | |
| `bmp_pressure` | Pa (float) | Barometric pressure |
| `acc_x`, `acc_y`, `acc_z` | raw LSB (int) | Non-Bosch variant (100 Hz) |

## Data Protection Note

Per the system design, the InfluxDB instance is hosted on the **care-giver / medical partner** side.
The ML inference server queries InfluxDB to fetch sensor data for each prediction cycle.

The connection is configured in the client `.env`:
```
INFLUXDB_URL=https://...
INFLUXDB_TOKEN=...
INFLUXDB_ORG=...
INFLUXDB_BUCKET=sensor_data
```

## Sampling Rates

| Sensor | Hardware Rate | Model Rate | Resampling |
|--------|-------------|-----------|-----------|
| Bosch ACC | 25 Hz | 50 Hz | Linear upsampling |
| Non-Bosch ACC | 100 Hz | 50 Hz | Decimation / averaging |
| Barometer | 25 Hz | 25 Hz | No resampling |

## InfluxDB Setup (for care-giver organisation)

If self-hosting InfluxDB (recommended for data protection):

```bash
# Using the infrastructure Docker Compose:
docker-compose -f infrastructure/docker-compose.yml up -d influxdb

# Or standalone:
docker run -p 8086:8086 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=yourpassword \
  -e DOCKER_INFLUXDB_INIT_ORG=falldetect \
  -e DOCKER_INFLUXDB_INIT_BUCKET=sensor_data \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=yourtoken \
  influxdb:2.7-alpine
```

Then configure the SmarKo app to upload to this InfluxDB instance.
