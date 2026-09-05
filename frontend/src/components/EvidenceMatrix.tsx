"use client";

import { motion } from "framer-motion";
import {
  ScanFace,
  Waves,
  Fingerprint,
  Timer,
  AudioLines,
  BarChart3,
} from "lucide-react";
import { ConsolidatedForensicReport } from "../lib/types";

interface EvidenceMatrixProps {
  report: ConsolidatedForensicReport;
}

interface Factor {
  name: string;
  icon: React.ElementType;
  score: number;
  status: "NATURAL" | "ANOMALOUS" | "UNCERTAIN";
  description: string;
}

function computeFactors(report: ConsolidatedForensicReport): Factor[] {
  const visual = report.modality_breakdown?.visual;
  const audio = report.modality_breakdown?.audio;
  const classical = report.modality_breakdown?.classical_forensics;

  // Use real scores from modality_breakdown; derive from raw_scores when available
  const visualFakeProb = visual?.raw_scores?.fake_prob
    ?? (report.metadata?.visual_spoof_prob as number | undefined)
    ?? visual?.confidence
    ?? null;
  const audioSpoofProb = audio?.raw_scores?.spoof_prob
    ?? (report.metadata?.audio_spoof_prob as number | undefined)
    ?? audio?.confidence
    ?? null;

  // Classical forensic cues — use real data only, never random
  const elaScore = classical?.ela_discrepancy_score ?? null;
  const fftScore = classical?.fft_spectral_anomaly ?? null;
  const boundaryScore = classical?.boundary_inconsistency ?? null;

  // Temporal coherence: derived from actual temporal_sync data
  const temporalAnomalyRate = report.temporal_sync.length > 0
    ? report.temporal_sync.filter((s) => s.is_anomaly).length /
      Math.max(report.temporal_sync.length, 1)
    : null;

  function toStatus(score: number | null): "NATURAL" | "ANOMALOUS" | "UNCERTAIN" {
    if (score === null) return "UNCERTAIN";
    if (score > 0.6) return "ANOMALOUS";
    if (score > 0.35) return "UNCERTAIN";
    return "NATURAL";
  }

  function safeScore(val: number | null): number {
    return val !== null ? Math.round(val * 100) : -1;
  }

  return [
    {
      name: "Facial Boundary Blending & Warping",
      icon: ScanFace,
      score: safeScore(boundaryScore ?? visualFakeProb),
      status: toStatus(boundaryScore ?? visualFakeProb),
      description: "Inter-frame blend boundary analysis detecting face-swap warping artifacts at skin-hair-background transitions.",
    },
    {
      name: "Fourier Frequency Spectrum Continuity",
      icon: Waves,
      score: safeScore(fftScore),
      status: toStatus(fftScore),
      description: "2D FFT power spectrum analysis revealing periodic lattice anomalies from GAN upsampling convolutions.",
    },
    {
      name: "Micro-Texture & Sensor Noise Distribution",
      icon: Fingerprint,
      score: safeScore(elaScore),
      status: toStatus(elaScore),
      description: "Spatial Rich Model (SRM) and Error Level Analysis (ELA) examining PRNU sensor noise consistency.",
    },
    {
      name: "Temporal Frame Coherence",
      icon: Timer,
      score: safeScore(temporalAnomalyRate),
      status: toStatus(temporalAnomalyRate),
      description: "Second-by-second aligned temporal consistency analysis across audio-visual modalities.",
    },
    {
      name: "Acoustic Resonances & Vocoder Phase Integrity",
      icon: AudioLines,
      score: safeScore(audioSpoofProb),
      status: toStatus(audioSpoofProb),
      description: "AASIST graph attention network analysis of spectral resonance patterns and STFT phase continuity.",
    },
  ];
}

const STATUS_CONFIG = {
  NATURAL: {
    label: "NATURAL",
    className: "status-natural",
    barClass: "progress-bar-fill-emerald",
  },
  ANOMALOUS: {
    label: "ANOMALOUS",
    className: "status-anomalous",
    barClass: "progress-bar-fill-crimson",
  },
  UNCERTAIN: {
    label: "UNCERTAIN",
    className: "status-uncertain",
    barClass: "progress-bar-fill-amber",
  },
};

export default function EvidenceMatrix({ report }: EvidenceMatrixProps) {
  const factors = computeFactors(report);

  return (
    <div className="glass-card p-6 space-y-5">
      <div className="flex items-center gap-2.5">
        <BarChart3 className="h-5 w-5 text-[#38bdf8]" />
        <h3 className="text-base font-bold text-white">
          Forensic Evidence Matrix
        </h3>
      </div>

      <div className="space-y-4">
        {factors.map((factor, idx) => {
          const Icon = factor.icon;
          const config = STATUS_CONFIG[factor.status];

          return (
            <motion.div
              key={factor.name}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.08 }}
              className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.04] space-y-2.5"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <Icon className="h-4 w-4 text-[#9ca3af]" />
                  <span className="text-xs font-semibold text-white">
                    {factor.name}
                  </span>
                </div>
                <div className="flex items-center gap-2.5">
                  <span className="font-mono text-xs font-bold text-white">
                    {factor.score >= 0 ? `${factor.score}%` : "N/A"}
                  </span>
                  <span
                    className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${config.className}`}
                  >
                    {factor.score >= 0 ? config.label : "DATA N/A"}
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="progress-bar-track h-1.5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(0, factor.score)}%` }}
                  transition={{
                    duration: 0.8,
                    delay: idx * 0.1,
                    ease: [0.16, 1, 0.3, 1],
                  }}
                  className={`h-full ${config.barClass}`}
                />
              </div>

              <p className="text-[11px] text-[#6b7280] leading-relaxed">
                {factor.description}
              </p>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
