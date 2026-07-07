const connectButton = document.getElementById("connectButton")
const resetButton = document.getElementById("resetButton")
const statusText = document.getElementById("statusText")
const flowText = document.getElementById("flowText")
const eventLog = document.getElementById("eventLog")
const remoteVideo = document.getElementById("remoteVideo")
const sceneSelect = document.getElementById("sceneSelect")

const LATENCY_ENABLED = new URLSearchParams(location.search).get('latency') === '1'

// Set after /api/scenes resolves; mirrors what the server reports as the
// initial scene. We keep it client-side so a failed switch can roll the
// dropdown back to whatever the server is actually on.
let activeScene = null

const allowedKeys = new Set([
  // PSM1 (left hand)
  "w", "a", "s", "d", "r", "f",
  "space", "c",
  "q", "e",
  // PSM2 (right hand)
  "arrowup", "arrowdown", "arrowleft", "arrowright",
  "pageup", "pagedown",
  ",", ".",
  ";", "'",
  // Modifier shared by both arms.
  "shift",
])
const activeKeys = new Set()

let peerConnection = null
let controlChannel = null
let connected = false

function logEvent(message) {
  const stamp = new Date().toLocaleTimeString()
  eventLog.textContent = `[${stamp}] ${message}\n${eventLog.textContent}`.slice(0, 5000)
}

function setStatus(message) {
  statusText.textContent = message
}

function setFlow(message) {
  flowText.textContent = message
}

function normalizeKey(rawKey) {
  const lower = String(rawKey || "").toLowerCase()
  // KeyboardEvent.key for spacebar is a literal " "; normalise to "space".
  return lower === " " ? "space" : lower
}

function sendControlAction(action) {
  if (!connected || !controlChannel || controlChannel.readyState !== "open") {
    return false
  }
  controlChannel.send(
    JSON.stringify({
      type: "action",
      action,
    })
  )
  setFlow(`sent ${action.event}${action.key ? `:${action.key}` : ""}`)
  return true
}

function enqueueAction(action) {
  const sent = sendControlAction(action)
  if (!sent) {
    setFlow(`not_sent ${action.event}${action.key ? `:${action.key}` : ""}`)
  }
}

function handleControlMessage(rawMessage) {
  let payload
  try {
    payload = JSON.parse(rawMessage)
  } catch (error) {
    logEvent(`invalid control payload: ${rawMessage}`)
    return
  }

  if (payload.type === "chunk_done") {
    logEvent(
      `chunk_done index=${payload.chunk_index}, frames=${payload.num_frames}, enqueued=${payload.enqueued_frames}`
    )
    return
  }

  if (payload.type === "reset_done") {
    logEvent("reset_done")
    setFlow("reset; render loop running")
    return
  }

  if (payload.type === "scene_set") {
    activeScene = payload.name
    if (sceneSelect.value !== payload.name) {
      sceneSelect.value = payload.name
    }
    logEvent(`scene_set name=${payload.name}`)
    setFlow(`scene=${payload.name}; render loop idle`)
    return
  }

  if (payload.type === "busy") {
    logEvent(`server busy: ${payload.message}`)
    return
  }

  if (payload.type === "error") {
    logEvent(`server error: ${payload.message}`)
    return
  }

  if (payload.type === 'frame_ts' && LATENCY_ENABLED) {
    const T_recv = performance.now()
    requestAnimationFrame(() => {
      const recv_to_raf_ms = performance.now() - T_recv
      if (controlChannel && controlChannel.readyState === 'open') {
        controlChannel.send(JSON.stringify({
          type: 'latency_echo',
          chunk_id: payload.chunk_id,
          recv_to_raf_ms,
        }))
      }
    })
    return
  }

  logEvent(`server message: ${rawMessage}`)
}

async function waitForIceGatheringComplete(pc) {
  if (pc.iceGatheringState === "complete") {
    return
  }
  await new Promise((resolve) => {
    const onStateChange = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", onStateChange)
        resolve()
      }
    }
    pc.addEventListener("icegatheringstatechange", onStateChange)
  })
}

async function connectSession() {
  connectButton.disabled = true
  setStatus("connecting")
  setFlow("creating peer connection")

  try {
    peerConnection = new RTCPeerConnection()
    controlChannel = peerConnection.createDataChannel("controls")
    peerConnection.addTransceiver("video", { direction: "recvonly" })

    controlChannel.onopen = () => {
      logEvent("control data channel open")
      setFlow("ready for action")
      sceneSelect.disabled = sceneSelect.options.length === 0
    }
    controlChannel.onclose = () => {
      logEvent("control data channel closed")
      setFlow("channel closed")
      sceneSelect.disabled = true
    }
    controlChannel.onmessage = (event) => {
      handleControlMessage(event.data)
    }

    peerConnection.ontrack = (event) => {
      const [stream] = event.streams
      if (stream) {
        remoteVideo.srcObject = stream
      }
    }

    peerConnection.onconnectionstatechange = () => {
      setStatus(peerConnection.connectionState)
      logEvent(`connection_state=${peerConnection.connectionState}`)
      if (peerConnection.connectionState === "connected") {
        connected = true
        resetButton.disabled = false
        setFlow("connected; press a key to start rendering")
      }
      if (["failed", "closed", "disconnected"].includes(peerConnection.connectionState)) {
        connected = false
        resetButton.disabled = true
      }
    }

    const offer = await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer)
    await waitForIceGatheringComplete(peerConnection)

    const response = await fetch("/api/webrtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(peerConnection.localDescription),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`offer failed (${response.status}): ${text}`)
    }
    const answer = await response.json()
    await peerConnection.setRemoteDescription(answer)
    logEvent("offer/answer completed")
  } catch (error) {
    setStatus("error")
    setFlow("failed")
    logEvent(`connect failed: ${error.message}`)
    connectButton.disabled = false
  }
}

function handleKeyDown(event) {
  if (!connected) {
    return
  }

  const key = normalizeKey(event.key)
  if (!allowedKeys.has(key)) {
    return
  }
  event.preventDefault()

  if (event.repeat) {
    return
  }
  activeKeys.add(key)
  enqueueAction({ event: "keydown", key })
}

function handleKeyUp(event) {
  if (!connected) {
    return
  }

  const key = normalizeKey(event.key)
  if (!allowedKeys.has(key)) {
    return
  }
  event.preventDefault()

  activeKeys.delete(key)
  enqueueAction({ event: "keyup", key })
}

function sendReset() {
  if (!connected || !controlChannel || controlChannel.readyState !== "open") {
    return
  }
  // Drop any locally-held key state — the server is wiping its own keyboard
  // state, and any keydown the browser thinks is held would re-fire on the
  // next 'step' anyway.
  activeKeys.clear()
  controlChannel.send(JSON.stringify({ type: "reset" }))
  setFlow("reset requested")
}

async function loadScenes() {
  try {
    const response = await fetch("/api/scenes")
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    const data = await response.json()
    const scenes = Array.isArray(data.scenes) ? data.scenes : []
    sceneSelect.innerHTML = ""
    for (const scene of scenes) {
      const option = document.createElement("option")
      option.value = scene.name
      option.textContent = scene.name
      sceneSelect.appendChild(option)
    }
    activeScene = data.active || (scenes[0] && scenes[0].name) || null
    if (activeScene) {
      sceneSelect.value = activeScene
    }
    // Dropdown stays disabled until the data channel is open — switching
    // scenes without a session is a no-op (the runtime starts on scenes[0]).
    sceneSelect.disabled = !(connected && controlChannel && controlChannel.readyState === "open")
  } catch (error) {
    sceneSelect.innerHTML = `<option value="">unavailable</option>`
    logEvent(`failed to load scenes: ${error.message}`)
  }
}

function sendSetScene(name) {
  if (!connected || !controlChannel || controlChannel.readyState !== "open") {
    return
  }
  // Drop locally-held key state for the same reason as reset — server's
  // about to wipe its keyboard_state.
  activeKeys.clear()
  controlChannel.send(JSON.stringify({ type: "set_scene", name }))
  setFlow(`set_scene ${name} requested`)
}

connectButton.addEventListener("click", () => {
  void connectSession()
})
resetButton.addEventListener("click", () => {
  sendReset()
})
sceneSelect.addEventListener("change", () => {
  const name = sceneSelect.value
  if (!name || name === activeScene) {
    return
  }
  if (!connected) {
    // Pre-connect: revert; the server starts on scenes[0] and there's no
    // way to change scene without an open data channel.
    sceneSelect.value = activeScene || ""
    return
  }
  sendSetScene(name)
})
window.addEventListener("keydown", handleKeyDown)
window.addEventListener("keyup", handleKeyUp)

void loadScenes()
