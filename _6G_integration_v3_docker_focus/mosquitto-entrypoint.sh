#!/bin/sh
set -e

if [ -z "$MQTT_USERNAME" ] || [ -z "$MQTT_PASSWORD" ]; then
    echo "ERROR: MQTT_USERNAME and MQTT_PASSWORD must be set in .env" >&2
    exit 1
fi

mosquitto_passwd -c -b /mosquitto/config/passwd "$MQTT_USERNAME" "$MQTT_PASSWORD"
echo "MQTT: password file generated for user '$MQTT_USERNAME'"

exec /usr/sbin/mosquitto -c /mosquitto/config/mosquitto.conf
