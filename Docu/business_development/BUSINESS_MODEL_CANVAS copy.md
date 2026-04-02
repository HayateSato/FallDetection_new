# Business Model Canvas
## RemoteGuard Health — Fall Detection & Continuous Vital Sign Monitoring Platform

*Company name is hypothetical. Sections marked (hypothetical) are ideas and suggestions based on
the technology in this repository combined with general knowledge of the digital health market.*

---

## The Product in One Sentence

A wearable + cloud platform that detects falls in real time, monitors PPG-derived vital signs
continuously, and routes alerts intelligently to patients, caregivers, and emergency responders —
reducing unnecessary emergency calls while ensuring no confirmed fall goes unattended.

---

## 1. Customer Segments

### Primary — B2B (main revenue driver)

| Segment | Who they are | Why they need this |
|---------|-------------|-------------------|
| **Residential care homes** | Operators of assisted living facilities (10–200 residents) | Staff reduction, liability reduction, CQC/regulatory compliance, incident documentation |
| **Home care agencies** | Organisations sending carers to elderly people's homes | Remote monitoring between visits, faster incident response, evidence for funding |
| **Hospital discharge teams** | NHS / private hospital wards managing post-surgical recovery at home | Reduce readmissions, monitor fragile patients in first 30–90 days post-discharge |
| **Health insurance companies** (hypothetical) | Insurers offering "active ageing" products | Reduces claims costs; attractive add-on for premium eldercare policies |

### Secondary — B2C (lower volume, high trust channel)

| Segment | Who they are | Why they need this |
|---------|-------------|-------------------|
| **Adult children of elderly parents** | 40–65 year olds with a parent living alone | Peace of mind, remote presence, avoid guilt of not knowing |
| **Self-monitoring elderly individuals** | Tech-comfortable 65+ who value independence | Stay at home longer, avoid nursing home entry |

### Tertiary — Enterprise / Research (hypothetical)

| Segment | Why they need this |
|---------|-------------------|
| **Pharmaceutical companies** running clinical trials | Objective ADL (Activities of Daily Living) data in real-world settings |
| **Medical device manufacturers** | OEM licensing of the fall detection + PPG algorithm stack |
| **Academic research institutions** | Anonymised dataset for ML research; co-development agreements |

---

## 2. Value Propositions

### For patients
- Wearable is unobtrusive (watch-form SmarKo device, worn on wrist)
- **10-second confirmation popup** — patient can dismiss false alarms themselves; no embarrassing unnecessary ambulance calls
- Continuous PPG monitoring catches silent risks (arrhythmia, SpO2 drop, elevated heart rate at rest) before a fall happens
- Independent living for longer — sensor-based safety net replaces need for constant human supervision

### For caregivers and family
- **Filtered alerts** — only notified when the patient confirmed a fall and needs help, or when they didn't respond (not on every detected movement)
- Full fall history dashboard with patient feedback columns (`user_fall`, `need_help`) — audit trail for insurance and incident reports
- Live patient list with fall count badges and vital sign trends
- SSE-based real-time push — no polling, instant alert on the browser

### For care home operators / system operators
- **Model comparison dashboard** — compare XGBoost model versions on the same CSV recording before deploying a new model
- Processing configuration (window size, resampling method, sensor type) adjustable at runtime without restart
- Prometheus + Grafana monitoring: latency, error rate, model drift detection via `ConfidenceDrift` alert
- Full audit trail in PostgreSQL (`api_request_log`, `inference_log`)
- `main.py` controllable from browser — no SSH needed for day-to-day operation

### Technical differentiators vs. simpler fall detectors
| Feature | Basic wrist alarm | Accelerometer-only pendant | **This platform** |
|---------|-------------------|--------------------------|-------------------|
| ML-based fall classification | No | Sometimes | **Yes — XGBoost, multi-model** |
| Patient confirmation before alert | No | No | **Yes — 10s popup flow** |
| PPG vital sign monitoring | No | No | **Yes — continuous** |
| Operator model management | No | No | **Yes — hot-swap, comparison** |
| Caregiver dashboard with history | No | Basic | **Full filter + feedback** |
| False positive suppression | No | No | **Confirmed by patient** |
| Retraining data capture | No | No | **Yes — user_fall labels stored** |

---

## 3. Channels

### Sales channels

| Channel | Stage | Notes |
|---------|-------|-------|
| **Direct B2B sales team** | Awareness → Contract | Care home groups, NHS ICBs, home care franchises |
| **Healthcare IT integrators** (hypothetical) | Awareness → Deployment | Companies that already sell nurse-call or EHR systems — white-label or referral agreement |
| **Medical device distributors** (hypothetical) | Awareness → Fulfillment | Regional distributors who already have relationships with care facilities |
| **Conference presence** (hypothetical) | Awareness | Care Show, HIMSS, Age UK events — demo the live dashboard |
| **Online (B2C)** (hypothetical) | Awareness → Purchase | Targeted ads to adult children; referrals from GPs and discharge nurses |

### Delivery channels

| Channel | What is delivered |
|---------|------------------|
| **Docker Compose (current)** | Full 10-service stack runs on a local server at the care facility or on a VPS |
| **Managed cloud SaaS** (hypothetical) | Company hosts the stack; customer just connects wearables and opens browser |
| **Hybrid** (hypothetical) | InfluxDB + real-time processing on-premise (latency-sensitive), dashboards and history in cloud |

---

## 4. Customer Relationships

| Relationship type | How |
|------------------|-----|
| **Dedicated onboarding** | 2–4 week implementation support: server setup, Alembic migration, Grafana dashboard walkthrough, staff training on caregiver and operator dashboards |
| **Self-service portal** (hypothetical) | HOW_TO_RUN.md expanded into a hosted knowledge base; video walkthroughs |
| **SLA-backed support tiers** (hypothetical) | Basic (email, 48h), Standard (email+chat, 8h), Enterprise (dedicated engineer, 1h critical) |
| **Model update service** (hypothetical) | Company pushes new XGBoost model versions to customers when drift is detected; operator approves switch via dashboard |
| **Quarterly analytics reports** (hypothetical) | PDF report per facility: fall rates by resident, model performance, PPG anomaly summary — value-add for care managers |
| **Research collaboration** (hypothetical) | Joint publications with academic partners using anonymised data; early access to algorithm updates in exchange |

---

## 5. Revenue Streams

### SaaS subscription (recurring — primary)

| Tier | Price (hypothetical) | Includes |
|------|---------------------|---------|
| **Starter** | £149/month per facility | Up to 10 residents, fall detection only, caregiver dashboard, email alerts |
| **Professional** | £349/month per facility | Up to 50 residents, fall + PPG monitoring, all dashboards, Grafana, phone alerts |
| **Enterprise** | Custom pricing | Unlimited residents, multi-site, SLA, dedicated support, EHR integration, custom model training |
| **B2C Family** | £19.99/month per device | Single patient, family caregiver dashboard, mobile app (hypothetical) |

### Hardware (one-time)

| Item | Price (hypothetical) | Notes |
|------|---------------------|-------|
| SmarKo wearable device | £250–£400 per unit | Sourced from SmarKo Health; resold with margin or direct referral |
| Gateway server (local install) | £500–£1,500 | Mini PC with Docker pre-configured; optional for cloud-only customers |

### Professional services (one-time)

| Service | Price (hypothetical) |
|---------|---------------------|
| Onboarding & setup | £500–£2,000 depending on facility size |
| EHR integration (Epic, Cerner, EMIS) | £3,000–£15,000 (hypothetical) |
| Custom model training on customer data | £5,000–£20,000 (hypothetical) |
| Staff training workshop (on-site) | £800/day (hypothetical) |

### Data & API licensing (hypothetical, future)

| Stream | Notes |
|--------|-------|
| Anonymised dataset licensing | To pharmaceutical companies / research institutions for gait + fall + PPG studies |
| Algorithm API licensing | OEM licensing of fall detection + PPG ML stack to device manufacturers |

---

## 6. Key Resources

### Intellectual property
- XGBoost fall detection models (v0, v3, v5_lsb) trained on SmarKo sensor data
- PPG signal processing pipeline (hypothetical — heart rate, SpO2, HRV, arrhythmia detection)
- Proprietary multi-stage alert logic (patient confirmation → conditional emergency escalation)
- Labelled dataset (`inference_log` with `user_fall` ground truth — grows with every deployment)

### Technology infrastructure
- Full-stack Docker Compose system: PostgreSQL, Redis, InfluxDB, nginx, Prometheus, Grafana, MinIO
- ml_server with hot-swap model switching, drift monitoring, and MinIO-based offline replay
- Two-channel Redis pub/sub architecture (false-positive-aware alerting)
- Patient SSE keepalive pattern (queue + background task — production-grade, not naive subscription)

### Human capital (hypothetical)
- ML engineers (model development, retraining pipeline, PPG algorithm R&D)
- Clinical informatics specialist (regulatory, validation studies, NHS/CQC compliance)
- Backend / DevOps engineers (platform reliability, security, EHR integration)
- Customer success team (onboarding, training, churn prevention)
- Regulatory affairs specialist (CE marking, UKCA, FDA 510k for medical device classification)

### Data assets
- Growing `inference_log` database — every deployment adds labelled sensor + fall + feedback data
- `feature_snapshot` table — feature vectors for every prediction (retraining source)
- PPG signal archive (hypothetical) — continuous vital sign time-series per patient

### Relationships
- SmarKo Health — hardware supply and integration partnership
- NHS / care home pilot partners — clinical validation and reference customers
- Cloud provider (hypothetical) — AWS or Azure for managed SaaS deployment

---

## 7. Key Activities

### Core (currently implemented)
- **Continuous model monitoring** — Prometheus `ConfidenceDrift` alert triggers when median fall confidence drops below 0.6 → retraining pipeline initiated
- **Platform maintenance** — Docker service health, Alembic migrations, Grafana dashboard updates
- **Algorithm development** — XGBoost model retraining using `user_fall=1` labelled rows from `inference_log` + feature vectors from `feature_snapshot`
- **Customer onboarding** — Docker setup, database migration, credential generation, dashboard walkthrough

### Required to scale (hypothetical)
- **PPG algorithm R&D** — developing validated heart rate, SpO2, HRV, and arrhythmia detection from raw photoplethysmography signal (requires clinical partner and validation dataset)
- **Regulatory certification** — CE / UKCA marking; if PPG-derived SpO2 or arrhythmia detection is marketed as medical-grade, Class IIa medical device classification likely required in UK/EU
- **EHR integration development** — HL7 FHIR APIs to push fall events and vital sign summaries into Epic, Cerner, EMIS
- **Mobile app development** — Push notifications for family caregivers (currently web-only)
- **Automated retraining pipeline** — `retrain.py` consuming `inference_log WHERE user_fall=1` + `feature_snapshot` → new model → operator approves deployment via dashboard
- **Multi-tenant SaaS infrastructure** — isolate customer data, per-tenant PostgreSQL schemas or databases, GDPR-compliant data residency

---

## 8. Key Partners

| Partner | Type | What they provide |
|---------|------|------------------|
| **SmarKo Health** | Hardware supplier | SmarKo wearable device (ACC 25Hz Bosch, barometer 25Hz, PPG sensor); InfluxDB cloud endpoint; ongoing sensor firmware support |
| **NHS Integrated Care Boards / Care homes** (hypothetical) | Pilot / validation partners | Clinical validation data, reference customers, letter of support for regulatory submission |
| **Cloud provider — AWS / Azure / GCP** (hypothetical) | Infrastructure | Managed Kubernetes or ECS for SaaS deployment; GDPR-compliant EU data regions |
| **EHR vendors — Epic / Cerner / EMIS** (hypothetical) | Integration partners | Certified app marketplace listings; FHIR API access for bi-directional data exchange |
| **Medical device distributors** (hypothetical) | Sales channel | Existing care facility relationships; regional coverage |
| **Academic research partners — University hospitals** (hypothetical) | R&D and validation | Clinical study design, IRB approval, publication co-authorship, access to hospital fall datasets |
| **Insurance companies — AXA Health, Vitality** (hypothetical) | Distribution + funding | Bundle device + subscription into "active ageing" insurance products; subsidised hardware |
| **AlertManager / notification infrastructure** (hypothetical) | Ops | SMTP relay (AWS SES), SMS gateway (Twilio), webhook endpoints for enterprise customers |

---

## 9. Cost Structure

### Fixed costs (monthly)

| Cost item | Notes |
|-----------|-------|
| **Engineering team salaries** (hypothetical) | 3–6 FTE: ML engineer, backend engineer, DevOps, clinical informatics |
| **Cloud infrastructure** (hypothetical) | PostgreSQL RDS, Redis ElastiCache, InfluxDB Cloud, S3/MinIO, nginx load balancer — scales with customers |
| **Regulatory affairs retainer** (hypothetical) | CE/UKCA ongoing compliance, especially as PPG features are added |
| **Cyber security audit** (hypothetical) | Annual penetration test, GDPR DPA registration, ISO 27001 preparation |

### Variable costs (per customer)

| Cost item | Notes |
|-----------|-------|
| **Cloud compute per facility** (hypothetical) | One Docker Compose stack per customer (self-hosted) or shared multi-tenant SaaS |
| **InfluxDB / PostgreSQL storage** | Grows with number of patients and prediction frequency |
| **Customer success / support time** | Onboarding + ongoing support; amortised over contract length |
| **SmarKo hardware COGS** | If reselling devices, hardware cost is ~60–70% of device sale price |

### One-time costs

| Cost item | Notes |
|-----------|-------|
| **Clinical validation study** (hypothetical) | £50,000–£200,000 to run a prospective study demonstrating sensitivity/specificity of fall detection in a real care environment |
| **Regulatory submission** (hypothetical) | £20,000–£80,000 for CE/UKCA technical file, notified body fees |
| **EHR integration development** (hypothetical) | £30,000–£100,000 per EHR system for certified integration |

---

## Strategic Notes & Suggestions

### Why the false-positive loop is a commercial differentiator
Most wearable fall detectors have a known problem: they trigger too many false alerts. Staff begin ignoring alerts ("alarm fatigue"). The two-stage patient confirmation flow in this system (`user_fall` + `need_help`) directly addresses this. **This is a sellable feature**, not just a technical detail — position it explicitly in sales materials.

### PPG as the upsell path (hypothetical)
The SmarKo hardware already contains a PPG sensor. Activating and processing PPG signals (heart rate, SpO2, HRV, atrial fibrillation screening) could be:
1. A premium tier add-on that increases ARPU significantly
2. A path to medical device classification (harder regulatory burden, but much higher defensibility and reimbursement potential via NICE/NHS technology appraisal)
3. A research data product — longitudinal PPG + fall correlation datasets are rare and valuable

### Ground truth as a moat (hypothetical)
Every deployment generates labelled data: `user_fall=1` rows are confirmed falls with full feature vectors in `feature_snapshot`. Over time, a company operating at scale accumulates a dataset that competitors cannot easily replicate. This is the basis for a model quality advantage that compounds with deployment scale.

### Regulatory path suggestion (hypothetical)
- **Phase 1 (now):** Market as a "wellness monitoring tool" — lower regulatory burden, faster to market
- **Phase 2:** Add clinical validation study; apply for UKCA Class IIa as a fall detection aid
- **Phase 3:** PPG arrhythmia screening → Class IIb; potential NHS reimbursement pathway via NICE MedTech Innovation Briefing

### Biggest risks
1. **Regulatory** — if PPG SpO2/arrhythmia detection is marketed as diagnostic, CE Class IIa/IIb required; delays time to market significantly
2. **Data privacy** — inference_log contains patient names and health data; GDPR Article 9 (special category health data); DPA registration and clinical governance framework required before enterprise sales
3. **Clinical adoption** — care staff need training; dashboards need to be genuinely easier than paper incident forms to achieve adoption
4. **Hardware dependency** — currently tightly coupled to SmarKo; a hardware API abstraction layer would reduce this risk
