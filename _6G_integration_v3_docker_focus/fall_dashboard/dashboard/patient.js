// Patient feedback popup — listens for falls via SSE, asks YES/NO.
//
// Flow:
//   1. SSE receives fall event with fall_id
//   2. "Did you fall?" popup (10s countdown)
//      YES → go to step 3
//      NO  → POST confirmed='no', done
//      timeout → POST confirmed='not_answered' (treated as fall)
//   3. "Do you need help?" popup (10s countdown)
//      YES → POST confirmed='yes_need_help', done
//      NO  → POST confirmed='yes_no_help', done
//      timeout → POST confirmed='yes_need_help' (assumed needs help)

const COUNTDOWN_SECONDS = 10;
const CIRCUMFERENCE = 2 * Math.PI * 34; // matches SVG r=34

let currentFallId = null;
let countdownInterval = null;
let eventSource = null;

// ---------------------------------------------------------------------------
// SSE connection
// ---------------------------------------------------------------------------
function connect() {
  eventSource = new EventSource('/api/stream');

  eventSource.addEventListener('connected', () => {
    document.getElementById('status-dot').className = 'dot dot-green';
    document.getElementById('status-text').textContent = 'Connected — monitoring';
  });

  eventSource.onmessage = (msg) => {
    document.getElementById('status-dot').className = 'dot dot-green';
    document.getElementById('status-text').textContent = 'Connected — monitoring';
    try {
      const event = JSON.parse(msg.data);
      if (event.fall_detected && event.fall_id) {
        showFallPopup(event);
      }
    } catch (e) {
      console.warn('Bad SSE message:', msg.data);
    }
  };

  eventSource.onerror = () => {
    document.getElementById('status-dot').className = 'dot dot-grey';
    document.getElementById('status-text').textContent = 'Disconnected — reconnecting...';
  };
}

// ---------------------------------------------------------------------------
// Fall popup (Question 1: Did you fall?)
// ---------------------------------------------------------------------------
function showFallPopup(event) {
  currentFallId = event.fall_id;
  closeAll();
  document.getElementById('popup-fall').classList.add('active');
  startCountdown('ring-fall', 'timer-fall', () => {
    // Timeout — no answer → treated as fall, go to help question
    answerFall('timeout');
  });
}

function answerFall(answer) {
  stopCountdown();
  closeAll();

  if (answer === 'no') {
    // Patient says they didn't fall → confirmed = 'no'
    sendConfirmation(currentFallId, 'no');
    showDone('No fall reported.', 'Response recorded. Stay safe.');
  } else {
    // YES or timeout → patient fell (or assumed fell), ask about help
    showHelpPopup();
  }
}

// ---------------------------------------------------------------------------
// Help popup (Question 2: Do you need help?)
// ---------------------------------------------------------------------------
function showHelpPopup() {
  document.getElementById('popup-help').classList.add('active');
  startCountdown('ring-help', 'timer-help', () => {
    // Timeout — no answer → assumed needs help
    answerHelp('timeout');
  });
}

function answerHelp(answer) {
  stopCountdown();
  closeAll();

  if (answer === 'no') {
    // Fell but doesn't need help
    sendConfirmation(currentFallId, 'yes');
    showDone('Fall confirmed.', 'No help requested. Take care.');
  } else {
    // YES or timeout → needs help (caregiver will be notified)
    sendConfirmation(currentFallId, 'yes');
    showDone('Fall confirmed — help is on the way.', 'A caregiver has been notified.');
  }
}

// ---------------------------------------------------------------------------
// Send confirmation to server
// ---------------------------------------------------------------------------
async function sendConfirmation(fallId, confirmed) {
  try {
    const resp = await fetch(`/api/falls/${fallId}/confirm?confirmed=${confirmed}`, {
      method: 'POST',
    });
    if (!resp.ok) {
      console.error('Confirmation failed:', resp.status, await resp.text());
    }
  } catch (e) {
    console.error('Could not send confirmation:', e);
  }
}

// ---------------------------------------------------------------------------
// Done screen (auto-dismiss after 5s)
// ---------------------------------------------------------------------------
function showDone(message, detail) {
  document.getElementById('done-message').textContent = message;
  document.getElementById('done-detail').textContent = detail;
  document.getElementById('popup-done').classList.add('active');
  setTimeout(() => {
    document.getElementById('popup-done').classList.remove('active');
    currentFallId = null;
  }, 5000);
}

// ---------------------------------------------------------------------------
// Countdown animation
// ---------------------------------------------------------------------------
function startCountdown(ringId, textId, onExpire) {
  const ring = document.getElementById(ringId);
  const text = document.getElementById(textId);
  let remaining = COUNTDOWN_SECONDS;

  ring.style.strokeDashoffset = '0';
  text.textContent = remaining;

  countdownInterval = setInterval(() => {
    remaining--;
    text.textContent = remaining;
    const offset = CIRCUMFERENCE * (1 - remaining / COUNTDOWN_SECONDS);
    ring.style.strokeDashoffset = offset;

    if (remaining <= 0) {
      stopCountdown();
      onExpire();
    }
  }, 1000);
}

function stopCountdown() {
  if (countdownInterval) {
    clearInterval(countdownInterval);
    countdownInterval = null;
  }
}

function closeAll() {
  document.getElementById('popup-fall').classList.remove('active');
  document.getElementById('popup-help').classList.remove('active');
  document.getElementById('popup-done').classList.remove('active');
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
connect();
