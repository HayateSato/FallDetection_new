# Email Template — Handover to FOCUS DevOps

---

**Subject:** Fall Detection — Deployment Package

---

Dear [FOCUS DevOps contact],

Please find attached the deployment package for the fall detection caregiver layer.

You will receive two files:
- **`_6G_Focus.zip`** — Helm chart and documentation
- **`caregiver_user_guide.docx`** — user guide for clinical staff (please forward to the caregivers who will use the dashboard)

We understand you previously requested that MCS host everything due to hardware constraints — we are happy to do so for the compute-heavy parts. However, we strongly recommend that FOCUS continues to host these two services, as they are the only ones that handle patient identifiers and fall history. Keeping them on your premises is important for data privacy and regulatory compliance. The services are intentionally lightweight and the deployment is straightforward.

After extracting the ZIP, please start with `README.md`. It walks through everything step by step.

Before deploying, you will need to decide on two subdomains on your Traefik cluster — one for the MQTT broker (the mobile app connects here) and one for the fall dashboard (caregivers open this in a browser). Once you have those, open `helm/values.yaml` and fill in the three `CHANGE_ME` fields: the two subdomains and your InfluxDB read token. The README (STEP 3) lists exactly which fields to change and which to leave as-is.

Registry credentials for pulling the Docker image will be shared by Mohammed separately.

Please let us know if you have any questions.

Best regards,
[Your name]
MCS Data Labs

