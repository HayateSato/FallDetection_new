# Business Model Canvas
## PulseResearch API — Advanced PPG-Derived Physiological Estimation for Research

*All sections are hypothetical / suggestions unless explicitly noted as derived from this repository.
This canvas describes a standalone research-focused business, distinct from the fall detection product.*

---

## The Product in One Sentence

A research API and SDK platform that provides validated, publication-grade algorithms for
estimating blood pressure, vascular stress, and autonomic nervous system state from raw PPG
waveforms — giving research teams the signal processing infrastructure so they can focus on
their clinical hypothesis, not the engineering.

---

## What Makes This "Research" and Not "Consumer"

This distinction shapes every element of the canvas:

| Consumer product | Research platform |
|-----------------|------------------|
| Single output (e.g. "your stress is high") | Full output: estimated value + confidence interval + waveform features + method reference |
| No validation required | Algorithm performance metrics mandatory (MAE, RMSE, Bland-Altman plots vs. reference device) |
| App store distribution | API/SDK with onboarding, documentation, IRB-compliant data handling |
| Volume pricing | Per-study or per-seat licensing |
| Regulatory: wellness | Regulatory: research use only (RUO) — not for diagnosis |
| User: general public | User: PhD students, clinical researchers, biomedical engineers, CRO scientists |

Research customers are not looking for a polished UX — they are looking for:
**correctness, reproducibility, documented methodology, and defensible outputs they can put in a paper.**

---

## 1. Customer Segments

### Primary — Academic Research

| Segment | Specific role | What they need |
|---------|--------------|----------------|
| **University biomedical engineering departments** | PhD students, postdocs running human subjects studies | Validated PPG algorithm they can cite; REST API or Python SDK they can call from their analysis pipeline |
| **Clinical research units** (hospital-affiliated) | Research nurses, clinical investigators | Easy signal ingestion, reliable BP/stress estimation without building signal processing from scratch |
| **Physiology & psychology departments** | Cognitive load, stress, emotion researchers | Continuous sympathetic nervous system proxy (HRV, LF/HF ratio, vascular resistance index) from wearable PPG |
| **Sports science / exercise physiology** | Performance researchers | Arterial stiffness, pulse wave velocity, autonomic recovery metrics during and after exercise |

### Secondary — Industry Research

| Segment | What they need |
|---------|---------------|
| **Pharmaceutical companies — Phase I/II trials** | Objective, continuous, non-invasive hemodynamic monitoring during drug effect studies; beats manual BP cuff at 15-minute intervals |
| **Contract Research Organisations (CROs)** | Algorithm-as-a-service they can drop into any trial using a PPG-capable wearable without internal signal processing expertise |
| **Medical device manufacturers (R&D division)** | Independent validation of their own cuffless BP / stress algorithm against a third-party reference before regulatory submission |
| **Wearable hardware companies** | Algorithm licensing to claim health feature on their device; co-development of PPG pipeline optimised for their sensor hardware |
| **Insurance / corporate wellness — research arm** | Research studies on stress biomarkers in workforce populations (not for individual pricing — aggregate anonymised research) |

### Tertiary — Government & Funding Bodies

| Segment | Notes |
|---------|-------|
| **NIHR / Innovate UK funded projects** | Often require open-source or consortium access to validated signal processing tools |
| **EU Horizon health research consortia** | Multi-site studies need a shared, standardised algorithm layer to ensure cross-site comparability |

---

## 2. Value Propositions

### For academic researchers

**"Stop building the pipeline. Start testing your hypothesis."**

A typical PhD student spends 3–6 months building and debugging a PPG preprocessing + feature extraction pipeline before they can even start their actual study. This platform eliminates that:

- Raw PPG in → cleaned, feature-extracted, physiological estimate out
- Every output includes: point estimate + 95% CI + signal quality index (SQI) + method reference (BibTeX-ready)
- Reproducible: same input → same output, versioned API endpoints (v1, v2 never deprecated mid-study)
- Publishable: algorithm methodology is documented to a level that satisfies peer review

### Core algorithm capabilities (hypothetical — research-grade)

| Algorithm | Input | Output | Validation benchmark |
|-----------|-------|--------|---------------------|
| **Cuffless blood pressure estimation** | PPG waveform (≥25Hz) + optional demographics (age, height, weight) | Systolic BP, Diastolic BP (mmHg) + confidence interval | AAMI SP10 standard: mean error <5 mmHg, SD <8 mmHg vs. reference cuff |
| **Vascular stress index** | Continuous PPG (≥1 min segment) | Sympathetic activation score (0–100), LF/HF HRV ratio, RMSSD | Validated against salivary cortisol and Trier Social Stress Test protocol |
| **Pulse Wave Velocity (PWV) proxy** | PPG + optional ECG (PTT calculation) | Estimated arterial stiffness index (m/s equivalent) | Compared against SphygmoCor reference device |
| **Augmentation Index (AIx)** | PPG waveform | AIx (%), vascular age estimate | Reference: invasive arterial line in controlled clinical setting |
| **Autonomic Nervous System (ANS) state** | Continuous PPG (5-min epoch) | HRV time-domain (RMSSD, SDNN, pNN50), frequency-domain (LF, HF, LF/HF), nonlinear (SD1/SD2) | Standard Task Force HRV guidelines (European Heart Journal 1996) |
| **Respiratory rate** | PPG | Breaths per minute ± CI | Validated against spirometry and capnography |

### For industry / CRO customers

- **Regulatory-ready documentation**: full algorithm description, validation study protocol, known limitations — usable as supporting material in FDA/CE submissions for the customer's own device
- **Signal Quality Index (SQI)**: every API call returns a quality score (0–1); results flagged as low-quality are explicitly labelled — protects research integrity
- **Waveform-agnostic preprocessing**: handles motion artefact, baseline wander, low perfusion — tested on SmarKo, Empatica E4, Polar OH1, Garmin, generic photodiode
- **GDPR-by-design**: no waveform data retained after processing unless customer explicitly enables research data sharing programme

---

## 3. Channels

| Channel | Who it reaches | Notes |
|---------|---------------|-------|
| **Developer documentation portal** (hypothetical) | PhD students, engineers | Primary discovery channel; OpenAPI spec, Python/R/MATLAB SDK, Jupyter notebook examples |
| **Academic conference presence** (hypothetical) | Researchers | Posters / workshops at IEEE EMBC, CinC, HIMSS, Pervasive Health; "tools and resources" sessions rather than commercial booths |
| **Research software registries** (hypothetical) | Researchers | Listed on Zenodo, JOSS (Journal of Open Source Software), bioRxiv preprint as citable tool |
| **Direct outreach to grant holders** (hypothetical) | PIs running funded studies | When a Horizon / NIHR / NIH grant mentions PPG or cuffless BP monitoring — direct contact offering trial access |
| **CRO partnerships** (hypothetical) | Industry studies | Pre-negotiated agreements with 3–5 CROs who include "PPG analysis" as a standard add-on service using this platform |
| **Hardware partner co-marketing** (hypothetical) | Device companies | Listed as "compatible algorithm provider" in SmarKo and similar hardware ecosystems |
| **Peer-reviewed publication** (hypothetical) | Academic community | Publishing the validation methodology in a journal (e.g. npj Digital Medicine, JMIR mHealth) is itself a marketing channel — researchers find and cite the algorithm |

---

## 4. Customer Relationships

| Relationship | How |
|-------------|-----|
| **Self-service API access** | Sign up, get API key, hit endpoints — works for small studies and exploratory use |
| **Study onboarding** (hypothetical) | For larger studies (>50 subjects, funded): 2–4 sessions with a research engineer to review study protocol, select appropriate algorithms, agree on epoch/windowing parameters — ensures the analysis is defensible |
| **Data review consultation** (hypothetical) | Research team sends a sample of waveforms; company returns a signal quality report + recommended preprocessing settings before the study begins — reduces post-collection surprises |
| **Versioned API contracts** (hypothetical) | Study registered at start date gets a locked algorithm version for the study duration — results remain reproducible even as the algorithm is improved; crucial for multi-year longitudinal studies |
| **Co-authorship / acknowledgement policy** (hypothetical) | Clear policy: if platform is used in a publication, minimum acknowledgement is required; co-authorship offered if substantial methodological contribution is made during study design |
| **Research data sharing opt-in** (hypothetical) | Customers can optionally contribute anonymised waveform + reference label data to a shared validation corpus in exchange for discounted access — builds the algorithm training dataset |

---

## 5. Revenue Streams

### API usage — primary model

| Tier | Price (hypothetical) | Includes |
|------|---------------------|---------|
| **Free tier (academic exploration)** | £0 / 500 API calls/month | All algorithms, SQI included, rate limited, no SLA — for students to prototype |
| **Study licence** | £800–£2,500 per study | Fixed number of subjects × duration; all algorithms; locked version; study report template; email support |
| **Institutional licence** | £4,000–£12,000/year | Unlimited studies within institution; priority support; optional on-premise deployment |
| **CRO / Industry per-trial** | £5,000–£25,000 per trial | Regulatory-grade documentation package; dedicated study manager; SLA 99.9%; data processing agreement |

### Algorithm licensing (hypothetical)

| Type | Price | Notes |
|------|-------|-------|
| **OEM algorithm licence** | £15,000–£80,000 upfront + royalty | Wearable hardware manufacturer embeds algorithm in device firmware or cloud; royalty per device shipped |
| **White-label SDK** | Custom | Hardware company ships SDK to their customers under their own brand |

### Professional services (hypothetical)

| Service | Price |
|---------|-------|
| Custom algorithm development | £10,000–£50,000 — e.g. adapting BP model for a specific population (paediatric, hypertensive, post-surgical) |
| Regulatory documentation package | £8,000–£20,000 — full technical file for CE/FDA submission supporting material |
| Signal quality audit | £1,500–£5,000 — review customer's existing PPG dataset and report on usability before study begins |
| Training workshop (online, 4h) | £500/session — "How to use PPG-derived metrics in clinical research" |

### Data consortium (hypothetical, long-term)

- Research partners contribute anonymised waveform + validated reference labels (BP cuff, Holter ECG)
- Consortium members receive early access to improved algorithm versions trained on larger dataset
- Revenue: membership fees (£2,000–£8,000/year per institution)
- Strategic value: builds proprietary training dataset that improves algorithm quality in a way competitors cannot match without similar clinical partnerships

---

## 6. Key Resources

### Algorithms and IP
- Validated signal preprocessing pipeline: bandpass filter, motion artefact removal (accelerometer-assisted or pure PPG), baseline wander correction, peak detection
- BP estimation model: likely a hybrid of feature engineering (PTT, pulse transit morphology, augmentation index) + ML (gradient boosting or deep learning) trained on large reference dataset
- HRV / ANS pipeline: standard frequency-domain and time-domain features, validated against published norms
- Signal Quality Index model: classifier that flags unreliable segments before they contaminate results
- Versioned, immutable algorithm releases (semantic versioning) — critical for research reproducibility

### Validation datasets
- Internal reference dataset: waveforms paired with gold-standard measurements (invasive arterial line, validated cuff device, ECG Holter)
- Diversity of the dataset matters: age range, ethnicities, skin tones (PPG is known to underperform on darker skin tones — an ethical and scientific imperative to address)
- Partnership with a hospital to collect reference data is essential and is the hardest resource to acquire

### Technical infrastructure
- REST API + Python / R / MATLAB SDK
- Batch processing endpoint (upload 1000-subject CSV → receive results CSV)
- Algorithm version registry: each release tagged, frozen, downloadable for offline reproducibility
- Audit log: every API call logs input hash, algorithm version, output hash — enables result reproducibility verification

### Human capital (hypothetical)
- Senior biomedical signal processing engineer (PhD-level; PPG, ECG, feature engineering)
- Clinical research coordinator (study design, IRB liaison, reference data collection)
- ML engineer (model training, validation, bias analysis across demographic groups)
- Research partnerships manager (academic relationships, conference presence, co-authorship)
- Regulatory affairs specialist (RUO classification, future IVD pathway if clinical use is pursued)

---

## 7. Key Activities

### Core platform activities
- **Algorithm development and validation**: continuous improvement of BP, stress, ANS, PWV models; each release documented with Bland-Altman analysis, RMSE, demographic breakdown
- **Signal quality research**: improving SQI classifier to handle a wider range of wearable hardware and motion conditions
- **API and SDK maintenance**: versioning, backward compatibility, documentation, SDK updates for new Python/R releases
- **Reference data collection** (hypothetical): ongoing partnership with clinical sites to collect waveform + gold-standard label pairs across diverse populations

### Research community activities
- **Study onboarding**: working with research teams to ensure their protocol is compatible with algorithm requirements
- **Peer-reviewed publication**: publishing validation methodology; this is both scientific duty and marketing
- **Conference workshops**: "hands-on PPG analysis" workshops at research conferences — build awareness and skills
- **Bias and fairness auditing** (hypothetical): regular audit of algorithm performance across ethnicity, age, BMI, skin tone — critical for academic credibility and ethical standing

### Business development activities
- **CRO partnership management**: maintaining pre-negotiated agreements and ensuring CRO researchers are trained
- **Grant ecosystem monitoring**: tracking NIHR, Horizon, NIH funding calls that mention PPG, cuffless BP, digital biomarkers — proactive outreach to new grant holders
- **Hardware partner integration**: ensuring preprocessing pipeline stays compatible with new firmware versions of partner hardware

---

## 8. Key Partners

| Partner | Role | What they provide |
|---------|------|-----------------|
| **Hospital clinical research units** (hypothetical) | Reference data collection | Waveform + invasive BP / Holter ECG paired datasets; IRB-approved collection protocols; clinical expertise for validation study design |
| **SmarKo Health** | Hardware | PPG-capable wearable with known sensor characteristics; integration and testing on actual hardware used by research customers |
| **Academic research groups — co-development** (hypothetical) | Algorithm validation and publication | Independent validation of algorithms; peer-reviewed publication; credibility in academic community |
| **CROs — ICON, Covance, PRA Health Sciences** (hypothetical) | Distribution into pharma trials | Pre-negotiated integration into standard trial toolkits; volume of studies |
| **IEEE / EMBC / Computing in Cardiology** (hypothetical) | Community presence | Conference sponsorship, workshop hosting, dataset challenge co-organisation |
| **MIMIC / PhysioNet / UK Biobank** (hypothetical) | Public dataset access | Training and benchmarking on large, publicly available PPG + reference datasets (MIMIC-III has arterial BP + PPG pairs) |
| **Regulatory consultancy** (hypothetical) | RUO and future IVD pathway | Ensuring "research use only" label is correctly applied; preparing for future clinical use reclassification |
| **Cloud provider — AWS / Azure** (hypothetical) | Infrastructure | GDPR-compliant EU compute; healthcare data handling agreements (BAA / GDPR DPA) |

---

## 9. Cost Structure

### Fixed costs

| Item | Notes |
|------|-------|
| **Senior biomedical engineer salary** (hypothetical) | Highest fixed cost; PhD-level, £60,000–£100,000/year |
| **Clinical research coordinator** (hypothetical) | Part-time or full-time depending on study pipeline; £35,000–£55,000/year |
| **Cloud infrastructure** (hypothetical) | API compute, storage for waveform processing, audit logs; scales with API volume; ~£500–£3,000/month at early stage |
| **Reference dataset collection** (hypothetical) | Ongoing cost of hospital partnership, equipment, participant compensation; £20,000–£80,000 per validation study |
| **Conference presence** (hypothetical) | 2–4 conferences/year; travel, registration, materials; ~£10,000–£25,000/year |

### Variable costs

| Item | Notes |
|------|-------|
| **API compute per call** | Waveform preprocessing is CPU-intensive; cost per analysis batch scales with usage |
| **Study onboarding time** | Research engineer time per study; partially recovered in study licence fee |
| **CRO trial support** | Dedicated study manager hours; recovered in trial fee |

### One-time investment costs (hypothetical)

| Item | Estimated cost |
|------|---------------|
| Initial validation study (BP + stress, N=200, hospital setting) | £60,000–£150,000 |
| API platform development (v1.0, SDK, documentation portal) | £40,000–£80,000 (or 6–12 months of engineering time) |
| Regulatory opinion (RUO classification + IVD pathway assessment) | £5,000–£15,000 |
| First peer-reviewed publication (methodology paper) | 6–12 months of research time; zero direct cost if done with academic partner |

---

## Strategic Notes & Suggestions

### Why "research only" is a deliberate strategic choice, not a limitation

Cuffless BP estimation marketed for clinical or consumer use immediately enters the medical device
regulatory pathway (Class IIa/IIb in EU; 510(k) in USA). That process takes 2–5 years and
£500,000+. The "research use only" label sidesteps this entirely — but only if the marketing is
consistent. Research customers understand and accept this distinction. This allows earlier
revenue and the chance to accumulate the clinical evidence that a future Class IIa submission
would require.

The strategy: **build the evidence base and the business simultaneously**.

### The skin tone / demographic fairness problem

PPG accuracy is known to be lower on darker skin tones due to melanin absorption effects on the
photoplethysmographic signal. Research published in 2021 (Bent et al., npj Digital Medicine)
showed significant performance gaps in consumer wearables across skin tones. For a
research-grade platform, this is not just an ethical issue — it is a scientific credibility issue.
Researchers in pharmacology and epidemiology work with diverse populations and will not
accept or cite a tool that has not reported stratified performance metrics.

**Suggestion:** Build demographic breakdown into every validation study from day one.
Report MAE by: age group, Fitzpatrick skin tone scale, BMI category, presence of hypertension.
This is differentiating versus consumer products that either do not report or report only headline
aggregate numbers.

### The MIMIC database as a starting point (hypothetical)

PhysioNet's MIMIC-III Waveform Database contains tens of thousands of simultaneous PPG and
invasive arterial blood pressure recordings from ICU patients. This is a freely available
gold-standard training and benchmarking dataset. A first BP estimation model can be developed
and benchmarked here before a prospective validation study is run. This reduces time and cost
to a working v1.0 algorithm significantly.

Limitation: MIMIC data is from critically ill, often sedated ICU patients — very different from
ambulatory wearable PPG. The model will need prospective validation in a walking, everyday-life
population before it is useful for research outside the ICU context.

### Key metric to watch: inter-subject vs. intra-subject accuracy

BP estimation from PPG can achieve good mean accuracy across a population while performing
poorly for specific individuals. Research customers running longitudinal studies (tracking one
person over time) will care more about intra-subject consistency than population-level MAE.
This is a distinct evaluation metric that should be explicitly reported — and is a research area
where the field is still maturing.

### Biggest risks specific to this business

| Risk | Mitigation |
|------|-----------|
| **Algorithm accuracy is not good enough for research use** | Be transparent about current limitations; provide SQI so bad data is flagged; publish limitations section; academic community respects honesty more than overclaiming |
| **"Research use only" label ignored by a customer** | Include in ToS that clinical or diagnostic use is prohibited; monitor for misuse in publications |
| **PPG-to-BP accuracy ceiling** (the science may be fundamentally limited without additional signals like ECG) | Diversify: stress / ANS metrics require only HRV from PPG and are more robust; BP remains a stretch goal |
| **Single point of failure: one validation dataset** | Partner with multiple clinical sites; pursue access to MIMIC, UK Biobank PPG subset, and a prospective study |
| **Academic customers expect open source** | Consider open-sourcing the preprocessing layer (bandpass filter, peak detection) while keeping the BP / stress models proprietary; this builds community trust and reduces support burden |
