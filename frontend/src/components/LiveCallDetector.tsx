"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Camera,
  CameraOff,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Zap,
  Activity,
  RotateCcw,
  Sliders,
  AlertTriangle,
  Maximize2,
  Minimize2,
  Radio,
  Eye,
  Scan,
} from "lucide-react";
import { LiveStreamTelemetry, LiveHistoryPoint } from "../lib/types";
import { WS_BASE_URL, sendLiveFrame, resetLiveStream } from "../lib/api";

export default function LiveCallDetector() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [useWebSocket, setUseWebSocket] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [simulateFake, setSimulateFake] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");

  const [telemetry, setTelemetry] = useState<LiveStreamTelemetry>({
    status: "idle",
    verdict: "UNCERTAIN",
    confidence: 0.5,
    fake_confidence: 0.5,
    real_confidence: 0.5,
    face_detected: false,
    bbox: null,
    threat_level: "NOMINAL",
    fps: 0,
    latency_ms: 0,
    history: [],
  });

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);

  // List available video devices
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then((devs) => {
      const videoDevs = devs.filter((d) => d.kind === "videoinput");
      setDevices(videoDevs);
      if (videoDevs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevs[0].deviceId);
      }
    }).catch(() => {});
  }, [selectedDeviceId]);

  // Session timer
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (isStreaming && sessionStartTime) {
      interval = setInterval(() => {
        setElapsedSec(Math.floor((Date.now() - sessionStartTime) / 1000));
      }, 1000);
    } else {
      setElapsedSec(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isStreaming, sessionStartTime]);

  // Setup WebSocket connection
  const setupWebSocket = useCallback(() => {
    if (!useWebSocket) return;
    try {
      const wsUrl = `${WS_BASE_URL}/ws/live-stream`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setWsConnected(true);
        setErrorMsg(null);
      };

      ws.onmessage = (event) => {
        try {
          const data: LiveStreamTelemetry = JSON.parse(event.data);
          setTelemetry((prev) => ({
            ...data,
            history: data.history && data.history.length > 0 ? data.history : prev.history,
          }));
        } catch {
          // ignore parsing error
        }
      };

      ws.onerror = () => {
        setWsConnected(false);
      };

      ws.onclose = () => {
        setWsConnected(false);
      };

      wsRef.current = ws;
    } catch (err: unknown) {
      setWsConnected(false);
      const msg = err instanceof Error ? err.message : "WebSocket connection failed";
      setErrorMsg(msg);
    }
  }, [useWebSocket]);

  // Capture frame and send
  const captureAndSendFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Fixed capture resolution for efficiency
    canvas.width = 480;
    canvas.height = 360;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.75);

    if (simulateFake) {
      // Simulation mode for demo
      const fakeScore = Math.min(0.96, 0.75 + Math.random() * 0.20);
      setTelemetry((prev) => ({
        ...prev,
        status: "active",
        verdict: "FAKE",
        confidence: fakeScore,
        fake_confidence: fakeScore,
        real_confidence: 1 - fakeScore,
        face_detected: true,
        bbox: prev.bbox || { x: 120, y: 80, w: 220, h: 220 },
        threat_level: "CRITICAL",
        fps: 15.0,
        latency_ms: 28.5,
        history: [
          ...prev.history.slice(-29),
          { t: Date.now() / 1000, score: fakeScore, verdict: "FAKE" },
        ],
      }));
      return;
    }

    if (useWebSocket && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          frame: dataUrl,
          timestamp: Date.now() / 1000,
        })
      );
    } else {
      // Fallback: REST post
      try {
        const result = await sendLiveFrame(dataUrl);
        setTelemetry((prev) => ({
          ...result,
          history: result.history && result.history.length > 0 ? result.history : prev.history,
        }));
      } catch {
        // throttled errors ignored
      }
    }
  }, [useWebSocket, simulateFake]);

  // Start Camera Stream
  const startStream = async () => {
    setErrorMsg(null);
    try {
      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsStreaming(true);
      setSessionStartTime(Date.now());
      setupWebSocket();

      // Capture at ~10 FPS
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = setInterval(captureAndSendFrame, 100);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Could not start camera feed";
      setErrorMsg(`Camera access failed: ${msg}. Check browser permissions.`);
    }
  };

  // Stop Camera Stream
  const stopStream = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setSessionStartTime(null);
  };

  // Reset telemetry
  const handleReset = async () => {
    await resetLiveStream().catch(() => {});
    setTelemetry({
      status: "idle",
      verdict: "UNCERTAIN",
      confidence: 0.5,
      fake_confidence: 0.5,
      real_confidence: 0.5,
      face_detected: false,
      bbox: null,
      threat_level: "NOMINAL",
      fps: 0,
      latency_ms: 0,
      history: [],
    });
  };

  // Draw cyber face reticle onto overlay canvas
  useEffect(() => {
    const overlay = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!overlay || !video || !isStreaming) return;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (telemetry.face_detected && telemetry.bbox) {
      // Scale from internal 480x360 coordinates to display video size
      const scaleX = overlay.width / 480;
      const scaleY = overlay.height / 360;

      const bx = telemetry.bbox.x * scaleX;
      const by = telemetry.bbox.y * scaleY;
      const bw = telemetry.bbox.w * scaleX;
      const bh = telemetry.bbox.h * scaleY;

      const isFake = telemetry.verdict === "FAKE";
      const isReal = telemetry.verdict === "REAL";
      const strokeColor = isFake ? "#ef4444" : isReal ? "#10b981" : "#f59e0b";

      // Glow box
      ctx.save();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);

      // Corner accents
      const corner = Math.min(24, bw / 4, bh / 4);
      ctx.lineWidth = 3;
      // Top-left
      ctx.beginPath();
      ctx.moveTo(bx, by + corner);
      ctx.lineTo(bx, by);
      ctx.lineTo(bx + corner, by);
      ctx.stroke();
      // Top-right
      ctx.beginPath();
      ctx.moveTo(bx + bw - corner, by);
      ctx.lineTo(bx + bw, by);
      ctx.lineTo(bx + bw, by + corner);
      ctx.stroke();
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(bx, by + bh - corner);
      ctx.lineTo(bx, by + bh);
      ctx.lineTo(bx + corner, by + bh);
      ctx.stroke();
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(bx + bw - corner, by + bh);
      ctx.lineTo(bx + bw, by + bh);
      ctx.lineTo(bx + bw, by + bh - corner);
      ctx.stroke();

      // Top Tag
      ctx.fillStyle = strokeColor;
      ctx.fillRect(bx, Math.max(0, by - 22), 120, 20);
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 10px monospace";
      ctx.fillText(
        isFake ? "🚨 SPOOF ALERT" : isReal ? "✓ HUMAN VERIFIED" : "SCANNING...",
        bx + 6,
        Math.max(14, by - 8)
      );
      ctx.restore();
    }
  }, [telemetry, isStreaming]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  const isFake = telemetry.verdict === "FAKE";
  const isReal = telemetry.verdict === "REAL";
  const isNotApplicable = telemetry.verdict === "NOT_APPLICABLE";

  return (
    <div className="w-full max-w-7xl mx-auto space-y-6">
      {/* ── Top Threat Banner (Only when active) ── */}
      <AnimatePresence>
        {isStreaming && isFake && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="flex items-center justify-between p-4 rounded-2xl bg-[#ef4444]/15 border border-[#ef4444]/40 shadow-[0_0_40px_rgba(239,68,68,0.25)]"
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-[#ef4444]/25 text-[#ef4444] animate-pulse">
                <ShieldAlert className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white tracking-wide">
                  CRITICAL BIOMETRIC ALERT: SYNTHETIC DEEPFAKE DETECTED
                </h3>
                <p className="text-xs text-[#fca5a5]">
                  Temporal inconsistency detected in facial landmarks and PRNU noise floor.
                  Confidence: {Math.round(telemetry.fake_confidence * 100)}%
                </p>
              </div>
            </div>
            <div className="px-3 py-1 rounded-full bg-[#ef4444]/30 border border-[#ef4444] text-[11px] font-mono text-white font-bold">
              THREAT LEVEL: HIGH
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Main Monitor Grid ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Video Call Screen (8 cols) */}
        <div className="lg:col-span-8 flex flex-col rounded-3xl border border-white/[0.08] bg-[#0c0d16] p-4 shadow-2xl relative overflow-hidden">
          {/* Top Status Bar */}
          <div className="flex items-center justify-between pb-3 border-b border-white/[0.06] mb-3">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span
                    className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${
                      isStreaming ? (isFake ? "bg-red-500" : "bg-emerald-500") : "bg-zinc-600"
                    }`}
                  />
                  <span
                    className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                      isStreaming ? (isFake ? "bg-red-500" : "bg-emerald-500") : "bg-zinc-600"
                    }`}
                  />
                </span>
                <span className="text-xs font-mono font-semibold tracking-wider text-white uppercase">
                  {isStreaming ? "SECURE CALL MONITOR ACTIVE" : "CAMERA STANDBY"}
                </span>
              </div>
              {isStreaming && (
                <span className="text-[11px] font-mono text-[#9ca3af] bg-white/[0.04] px-2 py-0.5 rounded-md">
                  {Math.floor(elapsedSec / 60)}:{(elapsedSec % 60).toString().padStart(2, "0")}
                </span>
              )}
            </div>

            <div className="flex items-center gap-3">
              {/* WS status badge */}
              <div className="flex items-center gap-1.5 text-[11px] font-mono text-[#9ca3af]">
                <Radio className={`w-3.5 h-3.5 ${wsConnected ? "text-[#10b981]" : "text-[#9ca3af]"}`} />
                <span>{useWebSocket ? (wsConnected ? "WS: Stream Active" : "WS: Connecting") : "REST Stream"}</span>
              </div>
            </div>
          </div>

          {/* Video Feed Canvas Area */}
          <div className="relative flex-1 min-h-[380px] bg-black/60 rounded-2xl overflow-hidden flex items-center justify-center border border-white/[0.04]">
            {/* Hidden capture canvas */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Video Element */}
            <video
              ref={videoRef}
              playsInline
              muted
              className={`w-full h-full object-cover ${isStreaming ? "block" : "hidden"}`}
            />

            {/* Reticle Overlay Canvas */}
            <canvas
              ref={overlayCanvasRef}
              className={`absolute inset-0 pointer-events-none z-10 w-full h-full ${
                isStreaming ? "block" : "hidden"
              }`}
            />

            {/* Placeholder when not streaming */}
            {!isStreaming && (
              <div className="flex flex-col items-center justify-center text-center p-8 space-y-4">
                <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.08] flex items-center justify-center text-[#38bdf8] shadow-[0_0_30px_rgba(56,189,248,0.15)]">
                  <Scan className="w-8 h-8" />
                </div>
                <div>
                  <h4 className="text-base font-bold text-white mb-1">
                    Live Webcam & Video Call Biometric Shield
                  </h4>
                  <p className="text-xs text-[#9ca3af] max-w-sm">
                    Activate camera to monitor live video calls or webcam streams with real-time
                    spatiotemporal deepfake detection and face tracking.
                  </p>
                </div>
                <button
                  onClick={startStream}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#38bdf8] to-[#6366f1] text-xs font-bold text-white shadow-[0_0_25px_rgba(56,189,248,0.3)] hover:scale-105 transition-transform"
                >
                  <Camera className="w-4 h-4" />
                  Start Live Camera
                </button>
              </div>
            )}

            {/* In-Video Cyber HUD Overlay (Only when streaming) */}
            {isStreaming && (
              <>
                {/* Top-Right Telemetry Card */}
                <div className="absolute top-3 right-3 z-20 flex flex-col gap-1.5 p-2.5 rounded-xl bg-black/70 backdrop-blur-md border border-white/[0.1] text-[10px] font-mono">
                  <div className="flex items-center justify-between gap-4 text-[#9ca3af]">
                    <span>STREAM FPS</span>
                    <span className="text-white font-bold">{telemetry.fps}</span>
                  </div>
                  <div className="flex items-center justify-between gap-4 text-[#9ca3af]">
                    <span>INFERENCE LATENCY</span>
                    <span className="text-[#38bdf8] font-bold">{telemetry.latency_ms}ms</span>
                  </div>
                  <div className="flex items-center justify-between gap-4 text-[#9ca3af]">
                    <span>FACE TRACKING</span>
                    <span className={telemetry.face_detected ? "text-[#10b981]" : "text-[#ef4444]"}>
                      {telemetry.face_detected ? "LOCKED" : "SEARCHING"}
                    </span>
                  </div>
                </div>

                {/* Bottom Left Verdict Tag */}
                <div className="absolute bottom-3 left-3 z-20 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-black/75 backdrop-blur-md border border-white/[0.1]">
                  {isFake ? (
                    <ShieldAlert className="w-4 h-4 text-[#ef4444]" />
                  ) : isReal ? (
                    <ShieldCheck className="w-4 h-4 text-[#10b981]" />
                  ) : (
                    <Shield className="w-4 h-4 text-[#f59e0b]" />
                  )}
                  <span
                    className={`text-xs font-bold font-mono ${
                      isFake ? "text-[#ef4444]" : isReal ? "text-[#10b981]" : "text-[#f59e0b]"
                    }`}
                  >
                    VERDICT: {telemetry.verdict}
                  </span>
                </div>
              </>
            )}
          </div>

          {/* Control Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-white/[0.06] mt-3">
            <div className="flex items-center gap-2">
              {isStreaming ? (
                <button
                  onClick={stopStream}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/20 border border-red-500/40 text-xs font-bold text-red-300 hover:bg-red-500/30 transition-colors"
                >
                  <CameraOff className="w-4 h-4" />
                  Stop Camera
                </button>
              ) : (
                <button
                  onClick={startStream}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-[#38bdf8]/20 border border-[#38bdf8]/40 text-xs font-bold text-[#38bdf8] hover:bg-[#38bdf8]/30 transition-colors"
                >
                  <Camera className="w-4 h-4" />
                  Start Camera
                </button>
              )}

              <button
                onClick={handleReset}
                className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/[0.04] border border-white/[0.08] text-xs text-[#9ca3af] hover:text-white transition-colors"
                title="Reset temporal tracking buffer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset
              </button>
            </div>

            {/* Device selector */}
            <div className="flex items-center gap-3">
              {devices.length > 1 && (
                <select
                  value={selectedDeviceId}
                  onChange={(e) => setSelectedDeviceId(e.target.value)}
                  className="px-2.5 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] text-xs text-[#9ca3af] focus:outline-none focus:border-[#38bdf8]"
                >
                  {devices.map((d, i) => (
                    <option key={d.deviceId || i} value={d.deviceId} className="bg-[#0c0d16] text-white">
                      {d.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
              )}

              {/* Simulation switch for demo */}
              <button
                onClick={() => setSimulateFake(!simulateFake)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-mono transition-colors border ${
                  simulateFake
                    ? "bg-red-500/20 border-red-500 text-red-400"
                    : "bg-white/[0.02] border-white/[0.08] text-[#9ca3af] hover:text-white"
                }`}
                title="Toggle simulated deepfake injection for presentation/testing"
              >
                <Zap className="w-3.5 h-3.5" />
                Simulate Deepfake: {simulateFake ? "ON" : "OFF"}
              </button>
            </div>
          </div>

          {errorMsg && (
            <div className="mt-3 p-2.5 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-400 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>

        {/* Right: Forensic Telemetry Deck (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          {/* Card 1: Live Biometric Gauge */}
          <div className="rounded-3xl border border-white/[0.08] bg-[#0c0d16] p-5 shadow-2xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-[#38bdf8]" />
                <h4 className="text-xs font-bold font-mono tracking-wider text-white uppercase">
                  Biometric Authenticity Gauge
                </h4>
              </div>
              <span className="text-[10px] font-mono text-[#9ca3af]">EMA α=0.82</span>
            </div>

            {/* Split Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono font-semibold">
                <span className="text-[#10b981]">
                  REAL: {Math.round(telemetry.real_confidence * 100)}%
                </span>
                <span className="text-[#ef4444]">
                  FAKE: {Math.round(telemetry.fake_confidence * 100)}%
                </span>
              </div>

              {/* Progress Track */}
              <div className="relative h-4 rounded-full bg-white/[0.04] p-0.5 overflow-hidden border border-white/[0.08]">
                <div
                  className="h-full rounded-full transition-all duration-300 ease-out"
                  style={{
                    width: `${Math.round(telemetry.fake_confidence * 100)}%`,
                    backgroundColor: isFake ? "#ef4444" : isReal ? "#10b981" : "#f59e0b",
                  }}
                />
              </div>

              {/* Threshold Labels */}
              <div className="flex justify-between text-[10px] font-mono text-[#6b7280]">
                <span>0% Authentic</span>
                <span className="text-[#10b981]">Real ≤ 30%</span>
                <span className="text-[#ef4444]">Fake ≥ 70%</span>
                <span>100% Synthetic</span>
              </div>
            </div>

            {/* Diagnostic Signals Pill Deck */}
            <div className="grid grid-cols-2 gap-2 pt-2 border-t border-white/[0.06]">
              <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                <div className="text-[10px] text-[#9ca3af] font-mono">FFT SPECTRUM</div>
                <div className="text-xs font-bold text-white mt-0.5">
                  {telemetry.visual_cues?.fft_spectral_anomaly !== undefined
                    ? `${(Number(telemetry.visual_cues.fft_spectral_anomaly) * 100).toFixed(1)}%`
                    : "0.0%"}
                </div>
              </div>
              <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                <div className="text-[10px] text-[#9ca3af] font-mono">BOUNDARY NOISE</div>
                <div className="text-xs font-bold text-white mt-0.5">
                  {telemetry.visual_cues?.boundary_inconsistency !== undefined
                    ? `${(Number(telemetry.visual_cues.boundary_inconsistency) * 100).toFixed(1)}%`
                    : "0.0%"}
                </div>
              </div>
            </div>
          </div>

          {/* Card 2: Temporal Confidence Waveform Sparkline */}
          <div className="rounded-3xl border border-white/[0.08] bg-[#0c0d16] p-5 shadow-2xl flex flex-col gap-3 flex-1">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-[#6366f1]" />
                <h4 className="text-xs font-bold font-mono tracking-wider text-white uppercase">
                  Temporal Waveform
                </h4>
              </div>
              <span className="text-[10px] font-mono text-[#9ca3af]">Last 30 samples</span>
            </div>

            {/* SVG Sparkline */}
            <div className="h-32 w-full bg-black/40 rounded-xl p-2 border border-white/[0.04] relative flex items-center">
              {telemetry.history.length >= 2 ? (
                <svg className="w-full h-full overflow-visible" preserveAspectRatio="none">
                  {/* Midline 50% */}
                  <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#374151" strokeDasharray="3,3" strokeWidth="1" />
                  {/* Danger line 70% */}
                  <line x1="0" y1="30%" x2="100%" y2="30%" stroke="#ef4444" strokeDasharray="2,2" strokeWidth="0.8" opacity="0.4" />
                  {/* Waveform polyline */}
                  <polyline
                    fill="none"
                    stroke={isFake ? "#ef4444" : isReal ? "#10b981" : "#f59e0b"}
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={telemetry.history
                      .map((pt, i, arr) => {
                        const x = (i / (arr.length - 1)) * 100;
                        const y = (1 - pt.score) * 100;
                        return `${x}%,${y}%`;
                      })
                      .join(" ")}
                  />
                </svg>
              ) : (
                <div className="w-full text-center text-xs font-mono text-[#6b7280]">
                  Awaiting frame history...
                </div>
              )}
            </div>

            <p className="text-[11px] text-[#9ca3af] leading-relaxed">
              Temporal hysteresis prevents video call frame jitter. Verdict flips to FAKE only when
              sustained biometric anomalies persist across sliding windows.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
