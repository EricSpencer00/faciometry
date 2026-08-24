/* Camera, overlay, and the two frame measurements a browser can honestly make.
 *
 * Subject distance and head pose dominate the error budget of every number
 * this system produces, and neither can be repaired after the shutter. The
 * overlay is therefore not decoration. The ring is a target for face size, the
 * horizon is a target for pupil height, and the readout carries the two
 * quantities a browser can actually measure from the stream: mean luminance
 * and a focus proxy.
 *
 * What it does not do is claim to measure distance. Estimating it would need a
 * face detector, and there is none here, so the ring is the instruction and
 * "1.5 m" is stated in words. An invented distance readout would be the same
 * mistake this project exists to avoid.
 *
 * The preview is mirrored because people cannot position themselves in an
 * unmirrored image. The captured frame is not, because the measurements
 * distinguish left from right.
 */

const METRIC_INTERVAL_MS = 200;
const PROBE_W = 160;
const PROBE_H = 120;

/* Mean luminance outside this band reads as under- or over-exposed. Chosen so
 * a correctly exposed face on a mid-grey ground sits in the middle. */
const LUMA_LOW = 62;
const LUMA_HIGH = 198;

/* Variance of the Laplacian below this is soft enough that landmark
 * localisation, and therefore every interval, degrades visibly. */
const FOCUS_FLOOR = 55;

export class Capture {
  constructor({ video, overlay, still, viewport, placeholder, onMetrics }) {
    this.video = video;
    this.overlay = overlay;
    this.still = still;
    this.viewport = viewport;
    this.placeholder = placeholder;
    this.onMetrics = onMetrics || (() => {});
    this.stream = null;
    this.blob = null;
    this.tiltDeg = null;
    this.raf = null;
    this.lastMetric = 0;
    this.probe = document.createElement('canvas');
    this.probe.width = PROBE_W;
    this.probe.height = PROBE_H;
    this.probeCtx = this.probe.getContext('2d', { willReadFrequently: true });
    this.phase = 0;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1440 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.viewport.dataset.state = 'live';
    this.placeholder.hidden = true;
    this.still.hidden = true;
    this.#listenForTilt();
    this.#loop();
  }

  stop() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.viewport.dataset.state = 'idle';
  }

  get live() {
    return Boolean(this.stream);
  }

  /* Draw the current frame at full sensor resolution and keep it as a blob.
   * Nothing leaves the tab here. */
  async capture() {
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    if (!w || !h) throw new Error('the camera has not produced a frame yet');
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    c.getContext('2d').drawImage(this.video, 0, 0, w, h);
    this.blob = await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.95));
    this.still.src = URL.createObjectURL(this.blob);
    this.still.hidden = false;
    this.viewport.dataset.state = 'held';
    this.stop();
    return this.blob;
  }

  /* Adopt a file the user picked instead of a camera frame. */
  async adopt(file) {
    this.stop();
    this.blob = file;
    this.still.src = URL.createObjectURL(file);
    this.still.hidden = false;
    this.placeholder.hidden = true;
    this.viewport.dataset.state = 'held';
    this.#clearOverlay();
    return file;
  }

  async retake() {
    this.blob = null;
    this.still.hidden = true;
    if (this.still.src) URL.revokeObjectURL(this.still.src);
    await this.start();
  }

  // ---------------------------------------------------------------- private

  #listenForTilt() {
    const handler = (e) => {
      if (e.gamma === null || e.gamma === undefined) return;
      /* Rotation about the viewing axis for a device held upright. Reported,
       * not corrected: the point is to let the operator level the camera
       * before the shutter, because image roll transfers almost one for one
       * into canthal tilt. */
      this.tiltDeg = e.gamma;
    };
    const DOE = window.DeviceOrientationEvent;
    if (!DOE) return;
    if (typeof DOE.requestPermission === 'function') {
      DOE.requestPermission()
        .then((state) => {
          if (state === 'granted') window.addEventListener('deviceorientation', handler);
        })
        .catch(() => {});
    } else {
      window.addEventListener('deviceorientation', handler);
    }
  }

  #loop = () => {
    this.raf = requestAnimationFrame(this.#loop);
    this.#draw();
    const now = performance.now();
    if (now - this.lastMetric > METRIC_INTERVAL_MS) {
      this.lastMetric = now;
      this.onMetrics(this.#measure());
    }
  };

  #clearOverlay() {
    const ctx = this.overlay.getContext('2d');
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
  }

  #draw() {
    const rect = this.viewport.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (this.overlay.width !== Math.round(rect.width * dpr)) {
      this.overlay.width = Math.round(rect.width * dpr);
      this.overlay.height = Math.round(rect.height * dpr);
    }
    const ctx = this.overlay.getContext('2d');
    const w = this.overlay.width;
    const h = this.overlay.height;
    ctx.clearRect(0, 0, w, h);

    const amber = 'rgba(224, 165, 60, 0.9)';
    const faint = 'rgba(224, 165, 60, 0.32)';
    ctx.lineWidth = Math.max(1, dpr);

    /* Face-size target. Filling it puts the head at roughly the framing the
     * clinical protocols ask for, which is the only lever on subject distance
     * available without a detector. */
    this.phase += 0.012;
    const breathe = 1 + Math.sin(this.phase) * 0.006;
    const cx = w / 2;
    const cy = h * 0.46;
    const rx = Math.min(w, h) * 0.21 * breathe;
    const ry = rx * 1.34;

    ctx.strokeStyle = amber;
    ctx.setLineDash([Math.round(10 * dpr), Math.round(8 * dpr)]);
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    /* Horizon. Both pupils belong on it; a head that is level makes it
     * disappear behind the eyes. Broken across the ring so it does not read as
     * a line through the face. */
    const eyeY = cy - ry * 0.24;
    ctx.strokeStyle = faint;
    ctx.beginPath();
    ctx.moveTo(0, eyeY);
    ctx.lineTo(cx - rx * 1.15, eyeY);
    ctx.moveTo(cx + rx * 1.15, eyeY);
    ctx.lineTo(w, eyeY);
    ctx.stroke();

    ctx.strokeStyle = amber;
    ctx.beginPath();
    ctx.moveTo(cx - rx * 0.98, eyeY);
    ctx.lineTo(cx - rx * 0.55, eyeY);
    ctx.moveTo(cx + rx * 0.55, eyeY);
    ctx.lineTo(cx + rx * 0.98, eyeY);
    ctx.stroke();

    /* Midline ticks, for yaw. */
    ctx.strokeStyle = faint;
    ctx.beginPath();
    for (let y = cy - ry; y < cy + ry; y += 14 * dpr) {
      ctx.moveTo(cx, y);
      ctx.lineTo(cx, y + 6 * dpr);
    }
    ctx.stroke();

    /* Frame corners. */
    const m = 14 * dpr;
    const len = 26 * dpr;
    ctx.strokeStyle = amber;
    ctx.beginPath();
    for (const [x, y, sx, sy] of [
      [m, m, 1, 1],
      [w - m, m, -1, 1],
      [m, h - m, 1, -1],
      [w - m, h - m, -1, -1],
    ]) {
      ctx.moveTo(x + sx * len, y);
      ctx.lineTo(x, y);
      ctx.lineTo(x, y + sy * len);
    }
    ctx.stroke();

    /* Spirit level, when the device reports one. */
    if (this.tiltDeg !== null) {
      const clamped = Math.max(-20, Math.min(20, this.tiltDeg));
      const level = Math.abs(clamped) < 2;
      ctx.strokeStyle = level ? amber : 'rgba(162, 54, 27, 0.95)';
      ctx.lineWidth = Math.max(1.5, dpr * 1.5);
      const span = rx * 1.5;
      const rad = (clamped * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(cx - Math.cos(rad) * span, h * 0.9 - Math.sin(rad) * span);
      ctx.lineTo(cx + Math.cos(rad) * span, h * 0.9 + Math.sin(rad) * span);
      ctx.stroke();
    }
  }

  /* Mean luminance and variance of the Laplacian over a downsampled frame.
   * Both are measured from the pixels rather than inferred, which is why they
   * are the only two numbers this overlay prints. */
  #measure() {
    if (!this.video.videoWidth) return {};
    this.probeCtx.drawImage(this.video, 0, 0, PROBE_W, PROBE_H);
    const { data } = this.probeCtx.getImageData(0, 0, PROBE_W, PROBE_H);
    const gray = new Float32Array(PROBE_W * PROBE_H);
    let sum = 0;
    for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
      const y = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
      gray[p] = y;
      sum += y;
    }
    const luma = sum / gray.length;

    let lapSum = 0;
    let lapSq = 0;
    let n = 0;
    for (let y = 1; y < PROBE_H - 1; y += 1) {
      for (let x = 1; x < PROBE_W - 1; x += 1) {
        const i = y * PROBE_W + x;
        const lap =
          4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - PROBE_W] - gray[i + PROBE_W];
        lapSum += lap;
        lapSq += lap * lap;
        n += 1;
      }
    }
    const focus = lapSq / n - (lapSum / n) ** 2;

    return {
      luma,
      exposure: luma < LUMA_LOW ? 'dark' : luma > LUMA_HIGH ? 'bright' : 'ok',
      focus,
      focusVerdict: focus < FOCUS_FLOOR ? 'soft' : 'ok',
      tiltDeg: this.tiltDeg,
    };
  }
}
