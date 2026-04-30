# Fundamental Questions — Partner Meeting
These 5 questions are the most load-bearing ones.
Each answer cascades clarity onto many smaller technical details automatically.
Ask these first — everything else follows from the answers.

---

## Q1. "Walk us through what happens today when a patient wearing the SmarKo device has a fall or a near-fall — where does the data go, and who or what currently sees it?"

**Why this is the most important question:**
This one open question reveals almost everything about their existing stack:
- Where is the sensor data stored? → gives us InfluxDB bucket/measurement/field names
- Does their system already poll InfluxDB? → tells us if they need our caregiver client or not
- Does anything currently detect or flag fall events? → tells us if we are replacing something or adding something new
- Who sees the result — a person, a dashboard, an automated system? → shapes how we need to deliver our output

*Listen carefully and map their answer to our architecture. Do not jump to solutions yet.*

---

## Q2. "Where do you want our fall detection result to arrive? In other words — what is the entry point of our output into your system?"

**The four realistic options (pick one or more):**

| Option | What it means | Good if |
|--------|--------------|---------|
| **Their FHIR server** | We POST an Observation directly to their FHIR endpoint | They already use FHIR as their data exchange format |
| **Their dashboard DB** | We return the result in our API response and they store it | They have their own backend that manages their dashboard data |
| **Back into InfluxDB** | We write a `fall_marker` field back to InfluxDB | Their dashboard already queries InfluxDB — zero change on their side |
| **Live SSE stream** | We push a real-time event to their dashboard browser | They want instant visual alerts with minimal backend work |

**Why this is fundamental:**
The answer to this question determines the entire integration architecture — which of our components they need, what format the output must be in, and what they need to build or change on their side.

*Our current implementation assumes FHIR, but all four options are achievable. Do not assume — ask directly.*

---

## Q3. "Do you have a FHIR server running in your current stack? Is FHIR format a hard requirement for this integration, or is it a preference?"

**Why this matters:**
We built FHIR R4 output because a colleague mentioned they use FHIR — but this was never confirmed directly. FHIR is just a JSON format wrapped around our result. If they do not need it, plain JSON is simpler and has no compatibility risk.

**What the answer unlocks:**
- Yes, FHIR required → confirm which server (HAPI / Azure / Firely), get the base URL and auth token
- No FHIR server → we can return plain JSON; simplifies Section 5 data contract entirely
- FHIR preferred but flexible → good to know — we can offer both and let them choose

---

## Q4. "Is your monitoring dashboard something we can connect to, or are you expecting us to provide the dashboard as well?"

In other words: **do they want to integrate our output into their existing dashboard, or do they want us to deliver a ready-made caregiver dashboard?**

**What the answer unlocks:**

| Their answer | Integration option | What we provide |
|-------------|-------------------|----------------|
| "We have a dashboard — connect to it" | Option A or B | Inference server + API/SSE endpoint they connect to |
| "We want a ready-made dashboard" | Option B | Full caregiver client with built-in dashboard |
| "We want it merged into our Docker stack" | Option C | Service definitions for their docker-compose |

This answer also tells us whether we need to ask about their frontend technology (React/Vue etc.) or not — only relevant if they are connecting their own dashboard to our stream.

---

## Q5. "What sample rate is configured for the SmarKo accelerometer in this trial — 25Hz or 100Hz?"

**Why this is a blocking technical detail:**
Our model was trained at 50Hz input (upsampled from 25Hz hardware). The SmarKo app can be configured for either 25Hz or 100Hz. If they are running at 100Hz, we downsample instead of upsample — this is a one-line change in our `.env`, but we must know before deployment.

**Why it is not obvious:**
The hardware supports both rates. Different trials may use different settings. The person to ask is whoever configured the SmarKo app for this trial — possibly a Charite researcher, not a FOCUS developer.

---

## How the answers connect to the rest of the agenda

```
Q1 answer  →  reveals InfluxDB structure, whether they need our poller (Section 3a + 4)
Q2 answer  →  determines entire output/integration architecture (Section 3c + 4 + 5)
Q3 answer  →  confirms or removes FHIR requirement (Section 3c + 5)
Q4 answer  →  determines which integration option (A/B/C) to propose (Section 4)
Q5 answer  →  unblocks configuration of inference server (Section 3a + 5)
```

Once you have answers to these five, most of the detailed checklist questions in the full agenda answer themselves.
