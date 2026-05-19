const enterVrButton = document.getElementById("enterVrButton")
const resetButton = document.getElementById("resetButton")
const wsText = document.getElementById("wsText")
const xrText = document.getElementById("xrText")
const rateText = document.getElementById("rateText")
const eventLog = document.getElementById("eventLog")
const remoteVideo = document.getElementById("remoteVideo")
const sceneSelect = document.getElementById("sceneSelect")

// Mirror of what the server reports as the active scene. Used to revert the
// dropdown if the user picks a scene the server rejects, or before the ws is
// up.
let activeScene = null

let ws = null
let xrSession = null
let xrRefSpace = null
let xrGl = null
// WebGL state for the MJPEG quad rendered in immersive view: shader program,
// vertex buffer, texture, and the attribute / uniform locations. Created in
// enterVr() after we have an xr-compatible GL context; cleared on session end.
let xrVideoQuad = null

// Previous-frame controller state per handedness. ``pos`` is play-space
// meters (xyz); ``quat`` is play-space orientation (xyzw). Both used to
// compute per-frame dpos / axis-angle drot. Cleared on every XR session
// start / end so the first frame after Enter VR emits zero deltas — and
// also reset implicitly each frame a controller drops out of tracking
// (we don't carry stale poses across tracking gaps).
const lastPose = {
  right: { pos: null, quat: null },
  left: { pos: null, quat: null },
}

function clearLastPose() {
  lastPose.right.pos = null
  lastPose.right.quat = null
  lastPose.left.pos = null
  lastPose.left.quat = null
}

// Rolling 1-second window for the "msg/s" HUD field. Counts successful
// WebSocket sends; reset whenever the window closes.
let sentCount = 0
let rateWindowStart = performance.now()

// Right-B-hold reset: 1 s. Tracks the press start (xrFrame timestamp in ms)
// and whether we've already fired the reset for this hold so a held button
// doesn't spam the server.
const RESET_HOLD_MS = 1000
let resetHoldStart = null
let resetSentForThisHold = false

// Left-Y-hold exit: 1 s. Same shape as the reset detector — ends the XR
// session and drops the user back to the 2D landing page.
const EXIT_HOLD_MS = 1000
let exitHoldStart = null
let exitFiredForThisHold = false

// Input semantics flags, fetched from the server's /vr_config endpoint on
// page load. Defaults match the historical absolute-frame behaviour so
// the page is still usable if the fetch fails.
//
// bodyRelativeTranslate=true  → rotate dpos by inverse headset yaw before
//                                sending. "Forward push" follows the user's
//                                heading rather than play-space.
// bodyRelativeRotation=true   → send conj(prev) * curr instead of
//                                curr * conj(prev) for drot. Rotation delta
//                                is expressed in the controller's previous
//                                frame, so wrist motion is invariant to
//                                controller orientation.
let bodyRelativeTranslate = false
let bodyRelativeRotation = false

// In-headset display panel sizing (metres). Defaults match the historical
// hardcoded quad — overridden when /vr_config returns a display section.
// Read by setupVideoQuad (size) and renderVideoQuad (distance).
const displayConfig = {
  widthM: 0.4,
  heightM: 0.3,
  distanceM: 1.2,
}

async function fetchVrConfig() {
  try {
    const resp = await fetch("/vr_config", { cache: "no-store" })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const cfg = await resp.json()
    bodyRelativeTranslate = !!cfg.body_relative_translate
    bodyRelativeRotation = !!cfg.body_relative_rotation
    if (cfg.display && typeof cfg.display === "object") {
      if (typeof cfg.display.width_m === "number") displayConfig.widthM = cfg.display.width_m
      if (typeof cfg.display.height_m === "number") displayConfig.heightM = cfg.display.height_m
      if (typeof cfg.display.distance_m === "number") displayConfig.distanceM = cfg.display.distance_m
    }
    logEvent(
      `vr_config: body_relative_translate=${bodyRelativeTranslate} ` +
      `body_relative_rotation=${bodyRelativeRotation} ` +
      `display=${displayConfig.widthM}x${displayConfig.heightM}m@${displayConfig.distanceM}m`,
    )
  } catch (e) {
    logEvent(`vr_config fetch failed (${e.message}) — using absolute defaults`)
  }
}

// Phase 1 uses WebSocket transport because Quest's network can't reach the
// server's WebRTC ICE candidates (NAT/proxy). See QUEST_PLAN.md for the
// path forward. WebRTC negotiation is deferred to Phase 3 along with video.

function logEvent(message) {
  const stamp = new Date().toLocaleTimeString()
  eventLog.textContent = `[${stamp}] ${message}\n${eventLog.textContent}`.slice(0, 5000)
}

function setWs(message) { wsText.textContent = message }
function setXr(message) { xrText.textContent = message }

// ---- Immersive-view video quad ----------------------------------------
// Renders the MJPEG <img> as a head-locked textured quad inside the XR
// framebuffer. DOM is hidden once the user enters immersive-vr; without
// this they'd see nothing in VR. Quad is 0.4 m × 0.3 m at 1.2 m — small
// "TV at arm's reach" so it doesn't dominate the FOV. Pattern lifted from
// quest3_tests/index.html (the POC's HUD), textured from <img> instead of
// a 2D canvas.

// Column-major 4×4 helpers. WebXR matrices arrive in this layout already,
// so all of our composition stays consistent without transposes.
function mat4Mul(a, b) {
  const o = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let s = 0
      for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k]
      o[col * 4 + row] = s
    }
  }
  return o
}

function mat4Translate(x, y, z) {
  const m = new Float32Array(16)
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1
  m[12] = x; m[13] = y; m[14] = z
  return m
}

function setupVideoQuad(gl) {
  const VS = `
    attribute vec2 a_pos;
    attribute vec2 a_uv;
    uniform mat4 u_mvp;
    varying vec2 v_uv;
    void main() {
      v_uv = a_uv;
      gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    }`
  const FS = `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_tex;
    void main() { gl_FragColor = texture2D(u_tex, v_uv); }`
  function sh(type, src) {
    const s = gl.createShader(type)
    gl.shaderSource(s, src)
    gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s))
    }
    return s
  }
  const prog = gl.createProgram()
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS))
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS))
  gl.linkProgram(prog)
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(prog))
  }
  // Quad size from displayConfig (set by /vr_config; defaults are the
  // historical 0.4 m × 0.3 m). UV (0,0) is the image top-left, so v=0
  // maps to the top edge of the quad (positive y). Flip the v
  // coordinates if the texture comes through upside-down. Aspect ratio
  // here should match the model output (runtime.resolution) — otherwise
  // the image gets stretched to fit.
  const halfW = 0.5 * displayConfig.widthM, halfH = 0.5 * displayConfig.heightM
  const verts = new Float32Array([
    -halfW, -halfH, 0, 1,
     halfW, -halfH, 1, 1,
    -halfW,  halfH, 0, 0,
     halfW,  halfH, 1, 0,
  ])
  const buf = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, buf)
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW)
  const tex = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  // 1×1 black placeholder so the quad is renderable before the first MJPEG
  // frame decodes.
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0,
    gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]),
  )
  return {
    prog, buf, tex,
    a_pos: gl.getAttribLocation(prog, "a_pos"),
    a_uv: gl.getAttribLocation(prog, "a_uv"),
    u_mvp: gl.getUniformLocation(prog, "u_mvp"),
    u_tex: gl.getUniformLocation(prog, "u_tex"),
  }
}

function uploadVideoTex(gl, quad) {
  if (!remoteVideo.complete || remoteVideo.naturalWidth === 0) return
  gl.bindTexture(gl.TEXTURE_2D, quad.tex)
  try {
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, remoteVideo,
    )
  } catch (e) {
    // Cross-origin or transient decode error — keep last good frame and
    // let the next iteration try again.
  }
}

function renderVideoQuad(gl, quad, viewerPose, layer) {
  // Head-locked: model = viewer_world * translate(0, 0, -distance) — quad
  // sits ``displayConfig.distanceM`` in front of the user and follows head
  // motion. The minus sign puts the quad in front along the camera's -z.
  const model = mat4Mul(
    viewerPose.transform.matrix,
    mat4Translate(0, 0, -displayConfig.distanceM),
  )
  gl.useProgram(quad.prog)
  gl.bindBuffer(gl.ARRAY_BUFFER, quad.buf)
  gl.enableVertexAttribArray(quad.a_pos)
  gl.vertexAttribPointer(quad.a_pos, 2, gl.FLOAT, false, 16, 0)
  gl.enableVertexAttribArray(quad.a_uv)
  gl.vertexAttribPointer(quad.a_uv, 2, gl.FLOAT, false, 16, 8)
  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D, quad.tex)
  gl.uniform1i(quad.u_tex, 0)
  for (const view of viewerPose.views) {
    const vp = layer.getViewport(view)
    gl.viewport(vp.x, vp.y, vp.width, vp.height)
    const viewMat = view.transform.inverse.matrix
    const mvp = mat4Mul(view.projectionMatrix, mat4Mul(viewMat, model))
    gl.uniformMatrix4fv(quad.u_mvp, false, mvp)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }
}

// ---- WebSocket --------------------------------------------------------

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws"
  ws = new WebSocket(`${proto}://${location.host}/ws`)
  setWs("connecting…")
  ws.onopen = () => {
    logEvent("ws connected")
    setWs("open")
    enterVrButton.disabled = !navigator.xr
    resetButton.disabled = false
    sceneSelect.disabled = sceneSelect.options.length === 0
    if (!navigator.xr) {
      setXr("WebXR unavailable")
    }
  }
  ws.onclose = () => {
    logEvent("ws closed, reconnecting in 2s")
    setWs("closed (retrying)")
    enterVrButton.disabled = true
    resetButton.disabled = true
    sceneSelect.disabled = true
    setTimeout(connectWs, 2000)
  }
  ws.onerror = () => logEvent("ws error")
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj))
    return true
  }
  return false
}

function sendReset() {
  if (send({ type: "reset" })) {
    logEvent("reset sent")
  } else {
    logEvent("reset not sent — ws not open")
  }
}

resetButton.addEventListener("click", () => sendReset())

async function fetchScenes() {
  try {
    const resp = await fetch("/scenes", { cache: "no-store" })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
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
    sceneSelect.disabled = !(ws && ws.readyState === WebSocket.OPEN)
    logEvent(`scenes loaded: ${scenes.map((s) => s.name).join(", ") || "(none)"}`)
  } catch (e) {
    sceneSelect.innerHTML = `<option value="">unavailable</option>`
    logEvent(`scenes fetch failed: ${e.message}`)
  }
}

sceneSelect.addEventListener("change", () => {
  const name = sceneSelect.value
  if (!name || name === activeScene) return
  if (!send({ type: "set_scene", name })) {
    sceneSelect.value = activeScene || ""
    logEvent("set_scene not sent — ws not open")
    return
  }
  activeScene = name
  logEvent(`set_scene ${name} sent`)
})

// Fire-and-forget: flags land in milliseconds; user has to click Enter VR
// anyway, so no race in practice. On failure we fall back to absolute mode.
fetchVrConfig()
fetchScenes()
connectWs()

// ---- WebXR session ----------------------------------------------------

async function enterVr() {
  if (xrSession) {
    await xrSession.end()
    return
  }
  if (!navigator.xr) {
    logEvent("WebXR not available")
    return
  }

  try {
    xrSession = await navigator.xr.requestSession("immersive-vr", {
      requiredFeatures: ["local-floor"],
    })
  } catch (e1) {
    try {
      xrSession = await navigator.xr.requestSession("immersive-vr", {
        optionalFeatures: ["local-floor"],
      })
    } catch (e2) {
      logEvent(`requestSession failed: ${e2.message}`)
      return
    }
  }
  send({ type: "session", action: "start" })
  logEvent("XR session started")
  enterVrButton.textContent = "Exit VR"
  setXr("running")

  // XRWebGLLayer needs a real xr-compatible GL context — we use it both to
  // satisfy the compositor and to render the MJPEG quad each frame.
  const canvas = document.createElement("canvas")
  xrGl = canvas.getContext("webgl", { xrCompatible: true, alpha: false })
  await xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, xrGl) })

  try {
    xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
  } catch {
    xrRefSpace = await xrSession.requestReferenceSpace("local")
  }

  try {
    xrVideoQuad = setupVideoQuad(xrGl)
  } catch (e) {
    logEvent(`video quad setup failed: ${e.message}`)
    xrVideoQuad = null
  }

  clearLastPose()
  resetHoldStart = null
  resetSentForThisHold = false
  exitHoldStart = null
  exitFiredForThisHold = false
  sentCount = 0
  rateWindowStart = performance.now()

  xrSession.addEventListener("end", () => {
    send({ type: "session", action: "end" })
    logEvent("XR session ended")
    xrSession = null
    xrGl = null
    xrRefSpace = null
    xrVideoQuad = null
    clearLastPose()
    enterVrButton.textContent = "Enter VR"
    setXr("idle")
  })

  xrSession.requestAnimationFrame(onXRFrame)
}

// ---- Per-frame controller read ----------------------------------------
// ``readArm(handedness, frame, session)`` returns ``{ dpos, drot, trigger,
// secondaryButton }`` for one controller, or ``null`` if that arm has no
// tracked grip-space pose. ``dpos``: per-frame play-space position delta
// (meters). ``drot``: axis-angle delta (3-vec, radians) of controller
// orientation — server multiplies by ``--rotate_scale`` and feeds as
// per-output-frame ω into the rot6d ramp. ``trigger``: analog
// ``gamepad.buttons[0].value`` in [0, 1]. ``secondaryButton``:
// ``gamepad.buttons[5].pressed`` (right = B, left = Y) — read by
// ``onXRFrame`` for the right-B-hold reset and left-Y-hold exit detectors.

// ---- Quaternion / yaw helpers -----------------------------------------
// All quaternions are xyzw layout. Hamilton-product convention.

// Unit-quaternion → axis-angle 3-vec (axis × angle, radians). Returns
// [0, 0, 0] near identity so a tracking blip doesn't emit a spurious delta.
function quatToAxisAngle(qx, qy, qz, qw) {
  if (qw < 0) { qx = -qx; qy = -qy; qz = -qz; qw = -qw }
  const vmag = Math.hypot(qx, qy, qz)
  if (vmag < 1e-8) return [0, 0, 0]
  const angle = 2 * Math.atan2(vmag, qw)
  const k = angle / vmag
  return [qx * k, qy * k, qz * k]
}

// World-frame rotation delta: q_delta = curr * conj(prev). The axis-angle
// is expressed in play-space — same axes for every frame. Wrist twist
// around the grip looks different depending on how the controller is
// oriented in the world.
function quatWorldDelta(prev, curr) {
  const px = -prev[0], py = -prev[1], pz = -prev[2], pw = prev[3]
  const cx = curr[0], cy = curr[1], cz = curr[2], cw = curr[3]
  const qx = cw * px + cx * pw + cy * pz - cz * py
  const qy = cw * py - cx * pz + cy * pw + cz * px
  const qz = cw * pz + cx * py - cy * px + cz * pw
  const qw = cw * pw - cx * px - cy * py - cz * pz
  return quatToAxisAngle(qx, qy, qz, qw)
}

// Body-frame rotation delta: q_delta = conj(prev) * curr. The axis-angle
// is expressed in the controller's previous frame, so "twist wrist around
// grip axis" always produces an axis-angle along the grip axis, regardless
// of where the controller currently points in play-space.
function quatBodyDelta(prev, curr) {
  const px = -prev[0], py = -prev[1], pz = -prev[2], pw = prev[3]
  const cx = curr[0], cy = curr[1], cz = curr[2], cw = curr[3]
  const qx = pw * cx + px * cw + py * cz - pz * cy
  const qy = pw * cy - px * cz + py * cw + pz * cx
  const qz = pw * cz + px * cy - py * cx + pz * cw
  const qw = pw * cw - px * cx - py * cy - pz * cz
  return quatToAxisAngle(qx, qy, qz, qw)
}

// Extract the yaw (rotation about world +y axis) from an xyzw unit
// quaternion. Returns radians. Standard ZYX-Euler yaw formula.
function extractYaw(qx, qy, qz, qw) {
  return Math.atan2(
    2 * (qw * qy + qx * qz),
    1 - 2 * (qy * qy + qx * qx),
  )
}

// Rotate a play-space vector by ``-yaw`` around +y, expressing it in the
// "headset-yawed" frame. With yaw=0 this is the identity, so it's safe to
// always call when the body-relative translate flag is on but the user
// hasn't turned yet.
function rotateByMinusYaw(v, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw)
  return [c * v[0] - s * v[2], v[1], s * v[0] + c * v[2]]
}

function readArm(handedness, frame, session, headsetYaw) {
  let src = null
  for (const s of session.inputSources) {
    if (s.handedness === handedness) { src = s; break }
  }
  if (!src || !src.gripSpace) return null
  const pose = frame.getPose(src.gripSpace, xrRefSpace)
  if (!pose) return null

  const slot = lastPose[handedness]

  const p = pose.transform.position
  const pos = [p.x, p.y, p.z]
  let dpos = [0, 0, 0]
  if (slot.pos) {
    dpos = [pos[0] - slot.pos[0], pos[1] - slot.pos[1], pos[2] - slot.pos[2]]
  }
  slot.pos = pos
  if (bodyRelativeTranslate) {
    // Re-express dpos in the user's heading frame. At yaw=0 (user facing
    // session-start direction) this is identity, so the empirical
    // server-side remap still applies the same way.
    dpos = rotateByMinusYaw(dpos, headsetYaw)
  }

  const o = pose.transform.orientation
  const quat = [o.x, o.y, o.z, o.w]
  let drot = [0, 0, 0]
  if (slot.quat) {
    drot = bodyRelativeRotation
      ? quatBodyDelta(slot.quat, quat)
      : quatWorldDelta(slot.quat, quat)
  }
  slot.quat = quat

  const gp = src.gamepad
  const trigger = (gp && gp.buttons && gp.buttons[0]) ? gp.buttons[0].value : 0
  const secondaryButton = !!(gp && gp.buttons && gp.buttons[5] && gp.buttons[5].pressed)
  return { dpos, drot, trigger, secondaryButton }
}

function updateRate() {
  const now = performance.now()
  const elapsed = (now - rateWindowStart) / 1000
  if (elapsed >= 1.0) {
    rateText.textContent = `${(sentCount / elapsed).toFixed(0)} msg/s`
    sentCount = 0
    rateWindowStart = now
  }
}

function exitVrFromController() {
  // Fire-and-forget. ``xrSession.end()`` resolves asynchronously and the
  // ``addEventListener("end", ...)`` handler set up in enterVr() does the
  // bookkeeping (clearLastPose, button labels, etc.).
  if (xrSession) {
    xrSession.end().catch((e) => logEvent(`xrSession.end failed: ${e.message}`))
  }
}

function onXRFrame(t, frame) {
  const session = frame.session
  session.requestAnimationFrame(onXRFrame)

  // Bind + clear the framebuffer each frame so the headset doesn't show
  // stale GPU memory, then draw the MJPEG quad on top.
  const layer = session.renderState.baseLayer
  xrGl.bindFramebuffer(xrGl.FRAMEBUFFER, layer.framebuffer)
  xrGl.clearColor(0, 0, 0, 1)
  xrGl.clear(xrGl.COLOR_BUFFER_BIT | xrGl.DEPTH_BUFFER_BIT)

  // Viewer pose is needed for the video quad AND for the headset-yaw the
  // body-relative-translate mode applies to dpos. Fetch once per frame.
  const viewerPose = frame.getViewerPose(xrRefSpace)
  if (xrVideoQuad) {
    uploadVideoTex(xrGl, xrVideoQuad)
    if (viewerPose) {
      renderVideoQuad(xrGl, xrVideoQuad, viewerPose, layer)
    }
  }

  let headsetYaw = 0
  if (bodyRelativeTranslate && viewerPose) {
    const q = viewerPose.transform.orientation
    headsetYaw = extractYaw(q.x, q.y, q.z, q.w)
  }

  const right = readArm("right", frame, session, headsetYaw)
  const left = readArm("left", frame, session, headsetYaw)

  // Right-B-hold reset detector. Fires once per hold; releases re-arm.
  if (right && right.secondaryButton) {
    if (resetHoldStart === null) {
      resetHoldStart = t
    } else if (!resetSentForThisHold && t - resetHoldStart >= RESET_HOLD_MS) {
      sendReset()
      resetSentForThisHold = true
      logEvent("reset triggered by right-B hold")
    }
  } else {
    resetHoldStart = null
    resetSentForThisHold = false
  }

  // Left-Y-hold exit detector. Mirrors the reset detector: 1 s hold ends
  // the XR session and drops the user back to the 2D landing page. After
  // firing we skip the remaining payload work for this frame — the session
  // is on its way out, so there's no point sending one last vr_input.
  if (left && left.secondaryButton) {
    if (exitHoldStart === null) {
      exitHoldStart = t
    } else if (!exitFiredForThisHold && t - exitHoldStart >= EXIT_HOLD_MS) {
      exitFiredForThisHold = true
      logEvent("exit triggered by left-Y hold")
      exitVrFromController()
      return
    }
  } else {
    exitHoldStart = null
    exitFiredForThisHold = false
  }

  // Skip the send entirely if neither controller is tracked — saves bw,
  // and server treats a missing arm section as zero motion anyway.
  if (right === null && left === null) return
  const payload = { type: "vr_input", t_ms: t }
  if (right) {
    payload.right = { dpos: right.dpos, drot: right.drot, trigger: right.trigger }
  }
  if (left) {
    payload.left = { dpos: left.dpos, drot: left.drot, trigger: left.trigger }
  }
  if (send(payload)) sentCount++
  updateRate()
}

enterVrButton.addEventListener("click", () => { void enterVr() })
