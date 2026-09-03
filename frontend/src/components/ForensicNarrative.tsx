"use client";

import { motion } from "framer-motion";
import {
  Sparkles,
  Video,
  Volume2,
  AlertTriangle,
  Lightbulb,
} from "lucide-react";
import { ConsolidatedForensicReport } from "../lib/types";

interface ForensicNarrativeProps {
  report: ConsolidatedForensicReport;
}

export default function ForensicNarrative({ report }: ForensicNarrativeProps) {
  const nl = report.natural_language_report;
  const isFake = report.verdict === "FAKE";
  const isReal = report.verdict === "REAL";
  const confidencePct = Math.round(report.overall_confidence * 100);

  // Generate a formal technical narrative if NLP report is not available
  const generateFallbackNarrative = (): string => {
    const mediaDesc = report.media_type.toLowerCase();
    const verdictDesc = isFake
      ? "exhibits characteristics consistent with synthetic generation or manipulation"
      : isReal
      ? "demonstrates characteristics consistent with authentic, unmanipulated media"
      : "presents ambiguous indicators that prevent a definitive classification";

    return `The submitted ${mediaDesc} specimen ${verdictDesc}. ` +
      `The multi-modal forensic analysis pipeline processed the specimen through ` +
      `${report.media_type === "AUDIO" ? "AASIST spectro-temporal graph attention" : "EfficientNet-B4 spatial feature encoding with multi-head temporal self-attention"} ` +
      `yielding a composite confidence score of ${confidencePct}%. ` +
      `${report.temporal_sync.length > 0
        ? `Temporal synchronization analysis across ${report.temporal_sync.length} time-aligned segments ${
            report.temporal_sync.filter((s) => s.is_anomaly).length > 0
              ? `identified ${report.temporal_sync.filter((s) => s.is_anomaly).length} anomalous segments requiring further investigation.`
              : "revealed consistent cross-modal alignment."
          }`
        : ""
      }`;
  };

  return (
    <div className="glass-card p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Sparkles className="h-5 w-5 text-[#38bdf8]" />
          <h3 className="text-base font-bold text-white">
            Formal Technical Narrative
          </h3>
        </div>
        <span className="text-[10px] font-mono text-[#6b7280] px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.06]">
          {nl?.provider_used || "Deterministic Engine"}
        </span>
      </div>

      {/* Executive Summary */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-4"
      >
        <h4 className="text-xs font-bold uppercase tracking-wider text-[#38bdf8] mb-2">
          Executive Summary
        </h4>
        <p className="text-sm text-[#e8eaed] leading-relaxed">
          {nl?.executive_summary || generateFallbackNarrative()}
        </p>
      </motion.div>

      {/* Visual Analysis */}
      {report.media_type !== "AUDIO" && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="space-y-2"
        >
          <h4 className="text-xs font-bold uppercase tracking-wider text-[#6366f1] flex items-center gap-1.5">
            <Video className="h-3.5 w-3.5" />
            Visual & Frequency Spectrum Analysis
          </h4>
          <p className="text-xs text-[#9ca3af] leading-relaxed">
            {nl?.visual_analysis_narrative ||
              "Visual analysis computed via EfficientNet-B4 spatial feature encoding with 2D FFT spectral decomposition and Error Level Analysis. Facial regions were aligned using YuNet ONNX with 20% bounding-box margin expansion."}
          </p>
        </motion.div>
      )}

      {/* Audio Analysis */}
      {report.media_type !== "IMAGE" && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="space-y-2"
        >
          <h4 className="text-xs font-bold uppercase tracking-wider text-[#10b981] flex items-center gap-1.5">
            <Volume2 className="h-3.5 w-3.5" />
            Acoustic & AASIST Analysis
          </h4>
          <p className="text-xs text-[#9ca3af] leading-relaxed">
            {nl?.audio_analysis_narrative ||
              "Audio subsystem performed AASIST graph attention network inference on the raw 16kHz waveform, analyzing spectro-temporal patterns for neural vocoder artifacts and phase discontinuities."}
          </p>
        </motion.div>
      )}

      {/* Temporal Notes */}
      {nl?.temporal_inconsistency_notes && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="space-y-2"
        >
          <h4 className="text-xs font-bold uppercase tracking-wider text-[#f59e0b] flex items-center gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5" />
            Temporal Inconsistency Notes
          </h4>
          <p className="text-xs text-[#9ca3af] leading-relaxed">
            {nl.temporal_inconsistency_notes}
          </p>
        </motion.div>
      )}

      {/* Forensic Recommendations */}
      {nl?.forensic_recommendations &&
        nl.forensic_recommendations.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="pt-4 border-t border-white/[0.06] space-y-2.5"
          >
            <h4 className="text-xs font-bold uppercase tracking-wider text-[#38bdf8] flex items-center gap-1.5">
              <Lightbulb className="h-3.5 w-3.5" />
              Actionable Recommendations
            </h4>
            <ul className="space-y-2">
              {nl.forensic_recommendations.map((rec, rIdx) => (
                <li
                  key={rIdx}
                  className="text-xs text-[#9ca3af] flex items-start gap-2.5"
                >
                  <span className="text-[#38bdf8] mt-0.5 shrink-0">▸</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
    </div>
  );
}
