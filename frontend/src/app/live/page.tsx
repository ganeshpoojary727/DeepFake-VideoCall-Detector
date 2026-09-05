"use client";

import { motion } from "framer-motion";
import {
  Shield,
  Video,
  Radio,
  Cpu,
  Lock,
  Layers,
  Sparkles,
} from "lucide-react";
import Navbar from "../../components/Navbar";
import Footer from "../../components/Footer";
import CustomCursor from "../../components/CustomCursor";
import LiveCallDetector from "../../components/LiveCallDetector";

export default function LiveDetectorPage() {
  return (
    <main className="relative min-h-screen bg-[#07080f] text-[#e8eaed]">
      <CustomCursor />

      {/* Ambient Grid & Background Glow */}
      <div className="pointer-events-none fixed inset-0 z-0 cyber-grid opacity-30" />
      <div className="pointer-events-none fixed top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[500px] bg-gradient-to-tr from-[#38bdf8]/10 via-[#6366f1]/10 to-transparent blur-[140px] rounded-full" />

      {/* Sticky Navigation */}
      <Navbar />

      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-8 pb-20 space-y-10">
        {/* Page Header */}
        <div className="space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.03] border border-white/[0.08] text-xs font-mono text-[#38bdf8]">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#ef4444] opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-[#ef4444]" />
            </span>
            <span>REAL-TIME STREAM GUARD — PHASE 5 ARCHITECTURE</span>
          </div>

          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
            <div>
              <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white">
                Live Video Call & Webcam Biometric Guard
              </h1>
              <p className="text-sm text-[#9ca3af] max-w-2xl mt-1">
                Zero-latency spatiotemporal deepfake interception engine. Continuously evaluates
                facial landmark jitter, PRNU sensor noise floor, and 2D FFT spectral energy in real-time.
              </p>
            </div>

            {/* Quick Specs Pill */}
            <div className="flex items-center gap-4 text-xs font-mono text-[#9ca3af] bg-white/[0.02] border border-white/[0.06] px-4 py-2 rounded-2xl">
              <div className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-[#38bdf8]" />
                <span>EffNet-B4 + Attn</span>
              </div>
              <div className="w-px h-3 bg-white/[0.1]" />
              <div className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-[#6366f1]" />
                <span>16-Frame Window</span>
              </div>
              <div className="w-px h-3 bg-white/[0.1]" />
              <div className="flex items-center gap-1.5">
                <Lock className="w-3.5 h-3.5 text-[#10b981]" />
                <span>Hysteresis Guard</span>
              </div>
            </div>
          </div>
        </div>

        {/* Live Detector Interface */}
        <LiveCallDetector />

        {/* Technical Architecture Deep Dive */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-6">
          <div className="p-6 rounded-3xl border border-white/[0.06] bg-white/[0.02] space-y-2">
            <div className="w-10 h-10 rounded-xl bg-[#38bdf8]/10 text-[#38bdf8] flex items-center justify-center mb-3">
              <Video className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Sliding Window Sampling
            </h3>
            <p className="text-xs text-[#9ca3af] leading-relaxed">
              Maintains an in-memory 16-frame circular buffer at 30 FPS. Captures spatiotemporal
              discrepancies such as face-swapping border shearing and eye-blinking anomalies.
            </p>
          </div>

          <div className="p-6 rounded-3xl border border-white/[0.06] bg-white/[0.02] space-y-2">
            <div className="w-10 h-10 rounded-xl bg-[#6366f1]/10 text-[#6366f1] flex items-center justify-center mb-3">
              <Radio className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Exponential Moving Average (EMA)
            </h3>
            <p className="text-xs text-[#9ca3af] leading-relaxed">
              Calculates smoothed temporal confidence using α=0.82 to prevent single-frame flickering
              during live Zoom or Microsoft Teams video conferences.
            </p>
          </div>

          <div className="p-6 rounded-3xl border border-white/[0.06] bg-white/[0.02] space-y-2">
            <div className="w-10 h-10 rounded-xl bg-[#10b981]/10 text-[#10b981] flex items-center justify-center mb-3">
              <Shield className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Dual-Threshold Hysteresis
            </h3>
            <p className="text-xs text-[#9ca3af] leading-relaxed">
              Verdicts only transition to FAKE when probability crosses ≥70% and return to REAL below ≤30%.
              Coin-flip scores (31%–69%) are stabilized as UNCERTAIN.
            </p>
          </div>
        </div>
      </div>

      <Footer />
    </main>
  );
}
