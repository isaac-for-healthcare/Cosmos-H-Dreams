const remoteVideo = document.getElementById("remoteVideo")
const videoText = document.getElementById("videoText")
const eventsText = document.getElementById("eventsText")
const eventLog = document.getElementById("eventLog")

const KNOWN_TYPES = new Set(["info", "headset", "session", "reset", "error"])
// Cap the DOM log so a long demo doesn't grow it unbounded.
const MAX_ENTRIES = 400

function setBadge(el, label, cls) {
  el.textContent = label
  el.className = `badge badge-${cls}`
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
