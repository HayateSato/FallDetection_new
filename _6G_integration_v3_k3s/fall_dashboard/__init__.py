"""Caregiver client for the 6G/Charite Fall Detection integration.

Polls InfluxDB for registered patients, calls the inference server,
stores fall history in its own database, and serves a minimal dashboard.
"""
