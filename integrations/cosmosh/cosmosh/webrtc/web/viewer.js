const remoteVideo = document.getElementById("remoteVideo")
const videoText = document.getElementById("videoText")
const eventsText = document.getElementById("eventsText")
const eventLog = document.getElementById("eventLog")
const driverBadge = document.getElementById("driverBadge")
const driverText = document.getElementById("driverText")
const sceneText = document.getElementById("sceneText")

const KNOWN_TYPES = new Set([
  "info", "headset", "session", "reset", "error", "driver", "scene",
])
// Cap the DOM log so a long demo doesn't grow it unbounded.
const MAX_ENTRIES = 400

function setBadge(el, label, cls) {
  el.textContent = label
  el.className = `badge badge-${cls}`
}

// Update the prominent "Active driver" badge from a driver-name string.
// Unknown values fall back to the neutral "unknown" style.
function setDriver(name) {
  const norm = (name || "").toLowerCase()
  const known = ["keyboard", "quest", "idle"]
  const cls = known.includes(norm) ? norm : "unknown"
  driverText.textContent = norm || "unknown"
  driverBadge.className = `badge driver-badge driver-${cls}`
}

function setScene(name) {
  sceneText.textContent = name || "—"
}

function timeStamp(tMs) {
  // Server t_ms is monotonic-since-process-start, so show wall time for
  // human readability instead. The server-side timestamp is still useful
  // for ordering if we later want it.
  void tMs
  const d = new Date()
  const hh = String(d.getHours()).padStart(2, "0")
  const mm = String(d.getMinutes()).padStart(2, "0")
  const ss = String(d.getSeconds()).padStart(2, "0")
  return `${hh}:${mm}:${ss}`
}

function appendEvent(evt) {
  const type = KNOWN_TYPES.has(evt.type) ? evt.type : "info"

  // Driver / scene events also update the top-bar badges. We still log
  // them in the activity feed so the timeline shows the transitions.
  if (evt.type === "driver") {
    setDriver(evt.message)
  } else if (evt.type === "scene") {
    // Server publishes a human-readable message like "Scene set to 'foo'.";
    // extract the scene name when we can, fall back to the raw message.
    const match = /'([^']+)'/.exec(String(evt.message ?? ""))
    setScene(match ? match[1] : evt.message)
  }

  const row = document.createElement("div")
  row.className = `evt evt-${type}`

  const time = document.createElement("span")
  time.className = "evt-time"
  time.textContent = timeStamp(evt.t_ms)

  const tag = document.createElement("span")
  tag.className = "evt-tag"
  tag.textContent = type

  const text = document.createElement("span")
  text.textContent = String(evt.message ?? "")

  row.appendChild(time)
  row.appendChild(tag)
  row.appendChild(text)

  // Only auto-scroll if the user hasn't scrolled away from the bottom — so a
  // spectator inspecting history isn't yanked back by new arrivals.
  const nearBottom =
    eventLog.scrollTop + eventLog.clientHeight >= eventLog.scrollHeight - 24
  eventLog.appendChild(row)
  while (eventLog.childElementCount > MAX_ENTRIES) {
    eventLog.removeChild(eventLog.firstElementChild)
  }
  if (nearBottom) eventLog.scrollTop = eventLog.scrollHeight
}

// ---- /video (MJPEG) ---------------------------------------------------
// The MJPEG <img> autoreconnects on its own when the server drops; we
// just reflect load/error state in the badge so spectators can tell at a
// glance whether the stream is live.

remoteVideo.addEventListener("load", () => setBadge(videoText, "video: live", "ok"))
remoteVideo.addEventListener("error", () => setBadge(videoText, "video: error", "err"))

// ---- Initial state ----------------------------------------------------
// Seeded once on page load so the topbar badges render before the SSE
// stream catches up. The unified server populates /admin/status; on the
// Quest-only server it 404s and we just keep the defaults.

async function loadInitialState() {
  try {
    const resp = await fetch("/admin/status", { cache: "no-store" })
    if (!resp.ok) return
    const data = await resp.json()
    if (data.driver) setDriver(data.driver)
    if (data.active_scene) setScene(data.active_scene)
  } catch {
    // Ignore — SSE will still populate the badges live.
  }
}
void loadInitialState()

// ---- /viewer_events (SSE) ---------------------------------------------
// EventSource handles reconnect/backoff natively; we just surface the
// open/error state and append received events. A trailing cache-buster
// avoids stale-connection reuse after a server restart.

const es = new EventSource(`/viewer_events?t=${Date.now()}`)
es.onopen = () => setBadge(eventsText, "events: live", "ok")
es.onerror = () => setBadge(eventsText, "events: retrying…", "err")
es.onmessage = (e) => {
  try {
    const evt = JSON.parse(e.data)
    appendEvent(evt)
  } catch {
    // Drop malformed payloads — server only emits well-formed JSON in normal
    // operation; one bad frame shouldn't break the log.
  }
}
