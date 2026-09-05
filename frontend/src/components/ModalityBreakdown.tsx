"use client";

import { motion } from "framer-motion";
import {
  Video,
  Mic,
  SlidersHorizontal,
  Info,
  Layers,
  VolumeX,
} from "lucide-react";
import { ConsolidatedForensicReport } from "../lib/types";

interface ModalityBreakdownProps {
  report: ConsolidatedForensicReport;
}

export default function ModalityBreakdown({ report }: ModalityBreakdownProps) {
  const visual = report.modality_breakdown?.visual;
  const audio = report.modality_breakdown?.audio;
  const metadata = report.metadata || {};

  // Extract visual spoof/fake probability
  const visualFakeProb =
    visual?.raw_scores?.fake_prob ??
    (typeof metadata.visual_spoof_prob === "number" ? metadata.visual_spoof_prob : null) ??
    (visual?.verdict === "FAKE" ? visual?.confidence : visual ? 1 - (visual?.confidence ?? 0.5) : null);

  // Extract audio spoof probability
  const audioSpoofProb =
    audio?.raw_scores?.spoof_prob ??
    (typeof metadata.audio_spoof_prob === "number" ? metadata.audio_spoof_prob : null) ??
    (audio?.verdict === "FAKE" ? audio?.confidence : audio ? 1 - (audio?.confidence ?? 0.5) : null);

  const hasVisual = typeof visualFakeProb === "number";
  const hasAudio = typeof audioSpoofProb === "number" && report.media_type !== "IMAGE";

  const visualVal = hasVisual ? visualFakeProb : 0;
  const audioVal = hasAudio ? audioSpoofProb : 0;

  const visualPct = hasVisual ? Math.round(visualVal * 100) : 0;
  const visualRealPct = 100 - visualPct;

  const audioPct = hasAudio ? Math.round(audioVal * 100) : 0;
  const audioRealPct = 100 - audioPct;

  const isVisualFake = hasVisual && visualVal >= 0.5;
  const isAudioFake = hasAudio && audioVal >= 0.5;

  // Gating rationale
  const gatingActive =
    hasVisual &&
    hasAudio &&
    (visualVal >= 0.5 || audioVal >= 0.5) &&
    Math.abs(visualVal - audioVal) >= 0.2;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="glass-card p-6 relative overflow-hidden space-y-6"
    >
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/[0.06] pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[#38bdf8]/20 to-[#6366f1]/20 border border-[#38bdf8]/30">
            <SlidersHorizontal className="h-5 w-5 text-[#38bdf8]" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              Modality Breakdown: Video vs Audio Analysis
            </h3>
            <p className="text-xs text-[#9ca3af] font-mono">
              Independent biometric analysis for visual facial stream & acoustic vocal stream
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] text-[#9ca3af]">
            Type: {report.media_type}
          </span>
          {gatingActive && (
            <span className="text-[11px] font-mono px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 font-semibold flex items-center gap-1">
              <Layers className="h-3 w-3" />
              Adversarial Gating Active
            </span>
          )}
        </div>
      </div>

      {/* Dual Stream Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Visual Modality Card */}
        <div
          className={`p-5 rounded-2xl border transition-all ${
            hasVisual
              ? isVisualFake
                ? "bg-red-950/10 border-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.08)]"
                : "bg-emerald-950/10 border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.08)]"
              : "bg-white/[0.02] border-white/[0.06]"
          }`}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2.5">
              <div
                className={`p-2 rounded-xl ${
                  isVisualFake
                    ? "bg-red-500/10 text-red-400 border border-red-500/20"
                    : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                }`}
              >
                <Video className="h-4 w-4" />
              </div>
              <div>
                <span className="text-xs uppercase tracking-wider font-mono text-[#9ca3af]">
                  Visual Stream
                </span>
                <h4 className="text-sm font-bold text-white">Video / Facial Track</h4>
              </div>
            </div>

            {hasVisual ? (
              <span
                className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded-full ${
                  isVisualFake
                    ? "bg-red-500/20 text-red-300 border border-red-500/40"
                    : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                }`}
              >
                {isVisualFake ? "MANIPULATED" : "AUTHENTIC"}
              </span>
            ) : (
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-white/[0.04] text-white/40">
                NO VIDEO
              </span>
            )}
          </div>

          {hasVisual ? (
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-extrabold font-mono text-white">
                  {visualPct}%
                  <span className="text-xs text-red-400 ml-1 font-normal">fake risk</span>
                </span>
                <span className="text-xs font-mono text-emerald-400">
                  {visualRealPct}% authentic
                </span>
              </div>

              {/* Progress Dual Bar */}
              <div className="h-2 w-full rounded-full bg-white/[0.06] overflow-hidden flex">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${visualRealPct}%` }}
                  transition={{ duration: 0.8 }}
                  className="h-full bg-gradient-to-r from-emerald-500 to-sky-400"
                />
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${visualPct}%` }}
                  transition={{ duration: 0.8 }}
                  className="h-full bg-gradient-to-r from-red-500 to-amber-500"
                />
              </div>

              <div className="text-[11px] text-[#9ca3af] font-mono pt-1 space-y-1">
                <div className="flex justify-between">
                  <span>Architecture:</span>
                  <span className="text-white/80">EfficientNet-B4 + Temporal Transformer</span>
                </div>
                <div className="flex justify-between">
                  <span>Classical Cues:</span>
                  <span className="text-white/80">ELA, 2D FFT, Boundary Laplacian</span>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-[#6b7280] italic py-3 font-mono">
              No video or image frames available in this media.
            </p>
          )}
        </div>

        {/* Audio Modality Card */}
        <div
          className={`p-5 rounded-2xl border transition-all ${
            hasAudio
              ? isAudioFake
                ? "bg-red-950/10 border-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.08)]"
                : "bg-emerald-950/10 border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.08)]"
              : "bg-white/[0.02] border-white/[0.06]"
          }`}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2.5">
              <div
                className={`p-2 rounded-xl ${
                  !hasAudio
                    ? "bg-white/[0.04] text-[#6b7280] border border-white/[0.06]"
                    : isAudioFake
                    ? "bg-red-500/10 text-red-400 border border-red-500/20"
                    : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                }`}
              >
                {!hasAudio ? <VolumeX className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
              </div>
              <div>
                <span className="text-xs uppercase tracking-wider font-mono text-[#9ca3af]">
                  Acoustic Stream
                </span>
                <h4 className="text-sm font-bold text-white">Audio / Vocal Track</h4>
              </div>
            </div>

            {hasAudio ? (
              <span
                className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded-full ${
                  isAudioFake
                    ? "bg-red-500/20 text-red-300 border border-red-500/40"
                    : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                }`}
              >
                {isAudioFake ? "SYNTHETIC / CLONED" : "AUTHENTIC VOICE"}
              </span>
            ) : (
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-white/[0.04] text-white/40">
                NO AUDIO TRACK
              </span>
            )}
          </div>

          {hasAudio ? (
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-extrabold font-mono text-white">
                  {audioPct}%
                  <span className="text-xs text-red-400 ml-1 font-normal">spoof risk</span>
                </span>
                <span className="text-xs font-mono text-emerald-400">
                  {audioRealPct}% authentic
                </span>
              </div>

              {/* Progress Dual Bar */}
              <div className="h-2 w-full rounded-full bg-white/[0.06] overflow-hidden flex">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${audioRealPct}%` }}
                  transition={{ duration: 0.8 }}
                  className="h-full bg-gradient-to-r from-emerald-500 to-sky-400"
                />
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${audioPct}%` }}
                  transition={{ duration: 0.8 }}
                  className="h-full bg-gradient-to-r from-red-500 to-amber-500"
                />
              </div>

              <div className="text-[11px] text-[#9ca3af] font-mono pt-1 space-y-1">
                <div className="flex justify-between">
                  <span>Architecture:</span>
                  <span className="text-white/80">AASIST (Graph Attention Network)</span>
                </div>
                <div className="flex justify-between">
                  <span>Front-End:</span>
                  <span className="text-white/80">Sinc-Convolution + Raw Waveform (16kHz)</span>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-[#6b7280] italic py-3 font-mono">
              {report.media_type === "IMAGE"
                ? "Static image file — no acoustic channel present."
                : "No audio track detected or video is silent."}
            </p>
          )}
        </div>
      </div>

      {/* Fusion Rationale Explanation Callout */}
      {hasVisual && hasAudio && (
        <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.06] flex items-start gap-3">
          <Info className="h-4 w-4 text-[#38bdf8] shrink-0 mt-0.5" />
          <div className="text-xs text-[#9ca3af] font-mono leading-relaxed">
            <span className="text-white font-semibold">How Fusion Evaluated This: </span>
            {isVisualFake && !isAudioFake ? (
              <span>
                <strong className="text-red-400">Deepfake Video with Authentic Audio.</strong>{" "}
                The visual track exhibits manipulation artifacts ({visualPct}% fake), while the voice track is original ({audioRealPct}% authentic). Under our adversarial gating rule, genuine audio is prevented from masking facial tampering.
              </span>
            ) : !isVisualFake && isAudioFake ? (
              <span>
                <strong className="text-red-400">Authentic Video with Voice-Clone Audio.</strong>{" "}
                The facial track appears genuine ({visualRealPct}% authentic), but the vocal track was identified as AI synthesized / cloned ({audioPct}% fake). Gating ensures the voice tampering triggers the final warning.
              </span>
            ) : isVisualFake && isAudioFake ? (
              <span>
                <strong className="text-red-400">Full Multimodal Deepfake.</strong>{" "}
                Both the visual track ({visualPct}% fake) and the audio track ({audioPct}% fake) display strong synthetic fingerprints.
              </span>
            ) : (
              <span>
                <strong className="text-emerald-400">Bilateral Authenticity Confirmed.</strong>{" "}
                Both visual biometric geometry ({visualRealPct}% authentic) and acoustic spectro-temporal resonance ({audioRealPct}% authentic) match physical recording patterns.
              </span>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
