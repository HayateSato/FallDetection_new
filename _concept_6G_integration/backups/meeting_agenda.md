# Meeting Agenda — Fall Detection Integration
<!-- **With:** FOCUS (technical partner)  
**Also involving:** Charite (medical trial operator)  
**Prepared by:** Hayate Sato (Data Scientist)  
**Duration suggested:** 60–90 minutes -->

<!-- --- -->

## Agenda Overview

| # | Topic | Time | Who leads |
|---|-------|------|-----------|
| 1 | Introductions + goal of meeting | 5 min | Both |
| 2 | What we deliver — system overview | 10 min | Us |
| 3 | Their existing stack — discovery | 20 min | Them |
| 4 | Integration boundary — where do the systems connect? | 15 min | Both |
| 5 | Data contract — exact inputs/outputs | 10 min | Both |
| 6 | Deployment — how and where does it run? | 10 min | Both |
| 7 | Open questions + next steps | 10 min | Both |

---

## 1. Introductions + Goal (5 min)

**Goal of this meeting:**
> Agree on *where* our fall detection system connects to FOCUS's existing infrastructure, and define the concrete next steps to build that connection.

**What success looks like after this meeting:**
- Both sides understand what the other provides
- Integration boundary is agreed (API call? shared DB? Docker container?)
- A checklist of information to exchange before development starts

---

## 2. What We Deliver — System Overview (10 min)

*You present this — use the partner_meeting_prep.md document as reference.*

Key points to cover:
- We provide a **REST API** that receives sensor data and returns a **FHIR R4 Observation**
- Fall detection is based on a **9-second sliding window** of accelerometer data
- We also provide a **caregiver client** that polls InfluxDB, calls our API, stores history, and pushes real-time alerts via Redis/SSE — but this is optional if they have equivalent components
- Model: XGBoost, 99.5% accuracy on test set, ~5ms inference, no GPU needed
- Configurable via `.env`; can run as Docker container

**Leave time for:** one or two clarifying questions from them before moving on.

---

## 3. Their Existing Stack — Discovery (20 min)

*This is the most important part of the meeting. Ask, listen, take notes.*

### 3a. Sensor + Data Storage
- [ ] What is the accelerometer sample rate? (We assume 25Hz Bosch)
- [ ] Is barometer data available? Do they want to use it?
- [ ] Where is raw sensor data stored? (InfluxDB?)
  - Bucket name? Measurement name?
  - What tag identifies each patient/device? (We use `macAddress`)
  - What field names for ACC axes? (We use `bosch_acc_x/y/z`)
- [ ] Is the InfluxDB cloud-hosted or on-premise?

### 3b. Dashboard + Monitoring System
- [ ] What does their current monitoring dashboard look like?
  - Just a patient list? Live alerts? Fall history?
- [ ] How is the dashboard built? (React/Vue/Angular/plain JS?)
- [ ] How does it currently receive real-time updates? (WebSocket? Polling? Server-Sent Events?)
- [ ] Do they have existing patient identifiers/IDs we should use?

### 3c. FHIR Setup
- [ ] Which FHIR server are they running? (HAPI FHIR, Azure, Firely, other?)
- [ ] Do they want us to **push** FHIR Observations automatically, or just **receive** them in our API response?
- [ ] What is the FHIR server base URL?
- [ ] What authentication does their FHIR server require? (Bearer token? Client certificate?)
- [ ] Are there specific FHIR resource types or coding systems they require?

### 3d. Infrastructure
- [ ] What is their deployment environment? (Cloud VM? On-premise server? Kubernetes?)
- [ ] What OS? (Linux strongly preferred for Docker)
- [ ] Do they already have Redis or a message broker in their stack?
- [ ] Do they already have Postgres, or do they need us to bring it?
- [ ] Are there firewall/network restrictions between their components?

---

## 4. Integration Boundary — Where Do the Systems Connect? (15 min)

*This is the key decision of the meeting. There are three options — agree on one.*

### Option A — Inference Server only (simplest)
```
Their system (existing InfluxDB poller + dashboard)
    │
    └──► POST /predict  (our inference server)
              │
              └──► FHIR Observation returned in response
                       │
                       └──► They push to their FHIR server + update dashboard
```
**We provide:** Docker container with inference server only  
**They do:** call our API from their existing code, handle FHIR push themselves  
**Requires:** they have or can build an InfluxDB poller on their side

### Option B — Full client + server (reference implementation)
```
InfluxDB (theirs)
    │
    └──► Our caregiver client (polls InfluxDB → calls inference → stores DB)
              │
              ├──► FHIR push to their server (optional)
              ├──► Redis fall event → their dashboard can subscribe
              └──► REST API /api/falls for fall history
```
**We provide:** both Docker containers (inference server + caregiver client)  
**They do:** point their dashboard at our SSE stream or REST API  
**Requires:** they expose InfluxDB credentials to our container

### Option C — Fully integrated into their Docker Compose
```
Their docker-compose.yml adds our services as additional containers
All services on the same Docker network
```
**We provide:** configuration for their docker-compose (not a separate stack)  
**They do:** merge our services into their deployment  
**Requires:** sharing their docker-compose structure with us

**Question to ask:** *"Which of these fits best with how your system is already structured?"*

---

## 5. Data Contract — Exact Inputs and Outputs (10 min)

*Make sure both sides agree on these specifics before leaving.*

### What we need as input (per polling cycle):
```
- Raw accelerometer arrays (X, Y, Z) — 15–30 seconds of data
- Timestamps in milliseconds
- Patient identifier (FHIR Patient/<id>)
- Sensor type (Bosch / non-Bosch)
- Sample rate (25 Hz / 100 Hz)
```

### What we return:
```
- FHIR R4 Observation (fall detected: true/false, confidence: 0.0–1.0)
- HTTP 200 always (errors return structured JSON, not HTTP 4xx/5xx spam)
```

### Questions to confirm:
- [ ] Should patient IDs follow FHIR format `Patient/001` or a different convention?
- [ ] Do they want barometer data included? (Requires `bmp_pressure` field in InfluxDB)
- [ ] Are there other fields they need in the FHIR Observation beyond fall status and confidence?
- [ ] Do they need the raw feature values returned for audit/logging?

---

## 6. Deployment — How and Where Does It Run? (10 min)

### Questions to resolve:
- [ ] Who manages deployment — us or them?
- [ ] Will we provide a Docker image (Docker Hub / registry), or do they build from source?
- [ ] How do they handle secrets/credentials? (Environment variables? Vault? `.env` file?)
- [ ] Do they need SSL/TLS for the inference server endpoint?
- [ ] What is the plan for updates/redeployment when we improve the model?
- [ ] Who is responsible for monitoring the service is running? (Health check endpoint exists at `GET /health`)

### MLOps question — who watches the model?

This needs to be agreed explicitly:

| Responsibility | Options |
|---------------|---------|
| Monitor inference server uptime | FOCUS DevOps (via their existing monitoring) or us |
| Monitor model accuracy over time | Us (data science) — but we need feedback labels from confirmed falls |
| Retrain model with new trial data | Us — but we need access to labeled fall data from the trial |
| Deploy model updates | Depends on deployment ownership above |

> **Important to raise:** We have a patient feedback mechanism (patient confirms yes/no on fall popup). If they want us to improve the model over time, they need a data sharing agreement with Charite for the labeled feedback data. This is a Charite/FOCUS/us governance question, not a technical one.

---

## 7. Open Questions + Next Steps (10 min)

### Immediate next steps — who does what before next contact:

**From FOCUS:**
- [ ] Share InfluxDB connection details (URL, token, bucket, measurement name, field names, tag names)
- [ ] Share FHIR server details (URL, auth method, required resource format)
- [ ] Share their docker-compose structure (or describe their deployment environment)
- [ ] Confirm which integration option (A, B, or C) they prefer
- [ ] Confirm patient ID format

**From us:**
- [ ] Provide our Docker image / docker-compose.yml
- [ ] Provide API documentation (Swagger UI auto-generated at `/docs`)
- [ ] Provide sample `/predict` request with real sensor data
- [ ] Provide `.env.example` with all required configuration variables
- [ ] Set up a test environment they can hit

**Governance (involves Charite):**
- [ ] Who owns the fall detection data? (Inference logs, FHIR Observations)
- [ ] What is the data retention policy?
- [ ] Is the patient feedback mechanism (yes/no on falls) in scope for this trial?
- [ ] What is the process if the model misfires during the trial?

---

## Things That Are Not Decided Yet — Be Honest About These

It is fine to say: *"I don't know yet, I will need to check and get back to you."*

| Topic | Why it is unclear | What to do |
|-------|------------------|------------|
| Non-SmarKo hardware support | Model trained on SmarKo data only; different hardware may need calibration | Ask for hardware specs + sample data |
| Real-world false positive rate | Test set was controlled; elderly population may differ | Plan a calibration/observation period at trial start |
| Their FHIR server compatibility | We implement FHIR R4 Observation, but some servers have quirks | Ask for their FHIR server docs / test endpoint |
| Scale (how many patients) | We tested with 1–2; concurrent polling untested at scale | Ask for expected patient count |
| Network architecture | We don't know if our containers can reach their InfluxDB | Ask for a network diagram or contact the DevOps person |
| SLA / uptime expectations | We don't have production SLA commitments yet | Agree on who monitors and what "acceptable downtime" means for a medical trial |

---

## Reference

Full technical details, model performance numbers, and expected Q&A are in:  
`_concept_6G_integration/partner_meeting_prep.md`
