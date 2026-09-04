"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Palette,
} from "lucide-react";

interface VerdictGaugeProps {
  verdict: "REAL" | "FAKE" | "UNCERTAIN" | "NOT_APPLICABLE" | string;
  confidence: number; // 0.0 to 1.0
  mediaType: string;
  processingTimeMs: number;
  contentCategory?: string | null;
}

export default function VerdictGauge({
  verdict,
  confidence,
  mediaType,
  processingTimeMs,
  contentCategory,
}: VerdictGaugeProps) {
  const [animatedConfidence, setAnimatedConfidence] = useState(0);

  const isFake = verdict === "FAKE";
  const isReal = verdict === "REAL";
  const isNotApplicable = verdict === "NOT_APPLICABLE";
  const confidencePct = Math.round(confidence * 100);

  const realPct = isReal ? confidencePct : 100 - confidencePct;
  const fakePct = isFake ? confidencePct : 100 - confidencePct;

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedConfidence(confidencePct), 100);
    return () => clearTimeout(timer);
  }, [confidencePct]);

  const _CATEGORY_LABELS: Record<string, string> = {
    DIGITAL_ART_ANIME: "Digital Illustration / Anime",
    SCENERY_OBJECT: "Scenery / Wallpaper / Object",
  };
  const categoryLabel = contentCategory
    ? (_CATEGORY_LABELS[contentCategory] ?? contentCategory)
    : "Non-Biometric Artwork";

  const verdictConfig = {
    REAL: {
      label: "AUTHENTIC MEDIA VERIFIED",
      icon: ShieldCheck,
      badgeClass: "verdict-real",
      textColor: "text-[#10b981]",
      glowClass: "glow-emerald",
      gradientClass: "gradient-text-verdict-real",
      bgGradient: "from-[#0c0d14] via-[#0a1a15] to-[#0c0d14]",
      meterGradient: "from-[#10b981] to-[#38bdf8]",
    },
    FAKE: {
      label: "SYNTHETIC DEEPFAKE DETECTED",
      icon: ShieldAlert,
      badgeClass: "verdict-fake",
      textColor: "text-[#ef4444]",
      glowClass: "glow-crimson",
      gradientClass: "gradient-text-verdict-fake",
      bgGradient: "from-[#0c0d14] via-[#1a0a0a] to-[#0c0d14]",
      meterGradient: "from-[#ef4444] to-[#f59e0b]",
    },
    UNCERTAIN: {
      label: "INCONCLUSIVE — MANUAL REVIEW RECOMMENDED",
      icon: AlertTriangle,
      badgeClass: "verdict-uncertain",
      textColor: "text-[#f59e0b]",
      glowClass: "glow-amber",
      gradientClass: "gradient-text-cyber",
      bgGradient: "from-[#0c0d14] via-[#1a1608] to-[#0c0d14]",
      meterGradient: "from-[#f59e0b] to-[#6366f1]",
    },
    NOT_APPLICABLE: {
      label: "NOT APPLICABLE — NON-BIOMETRIC CONTENT",
      icon: Palette,
      badgeClass: "verdict-uncertain",
      textColor: "text-[#a78bfa]",
      glowClass: "glow-amber",
      gradientClass: "gradient-text-cyber",
      bgGradient: "from-[#0c0d14] via-[#110d1a] to-[#0c0d14]",
      meterGradient: "from-[#a78bfa] to-[#6366f1]",
    },
  };

  const config =
    verdictConfig[verdict as keyof typeof verdictConfig] ??
    verdictConfig.UNCERTAIN;
  const VerdictIcon = config.icon;

  return (
    <div
      className={`glass-card p-6 md:p-8 relative overflow-hidden bg-gradient-to-r ${config.bgGradient}`}
    >
      {/* Subtle animated gradient behind */}
      <div className="absolute inset-0 opacity-20 pointer-events-none">
        <div className="absolute top-0 right-0 w-64 h-64 rounded-full blur-[100px] bg-gradient-to-br from-current to-transparent" />
      </div>

      <div className="relative z-10">
        {/* Verdict Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            {/* Verdict Icon */}
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ type: "spring", bounce: 0.4, duration: 0.8 }}
              className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl border ${config.badgeClass}`}
            >
              <VerdictIcon className="h-8 w-8" />
            </motion.div>

            <div>
              <motion.h2
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className={`text-xl md:text-2xl font-extrabold tracking-tight ${config.glowClass} text-white`}
              >
                {config.label}
              </motion.h2>
              {isNotApplicable ? (
                <p className="mt-1 text-xs text-[#a78bfa]/80 font-mono">
                  Content Category: {categoryLabel} • {mediaType} • Stage-0 Pre-Classifier
                </p>
              ) : (
                <p className="mt-1 text-xs text-[#9ca3af] font-mono">
                  Latency: {processingTimeMs.toFixed(0)}ms • {mediaType} •
                  Multi-Modal Fusion Engine
                </p>
              )}
            </div>
          </div>

          {/* Confidence Number */}
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3, type: "spring" }}
            className="text-right"
          >
            <div
              className={`text-4xl md:text-5xl font-extrabold tracking-tight ${config.gradientClass}`}
            >
              {isNotApplicable ? "N/A" : `${animatedConfidence}%`}
            </div>
            <span className="text-[11px] uppercase tracking-wider text-[#6b7280] font-mono">
              {isReal
                ? "Authenticity"
                : isFake
                ? "Deepfake Risk"
                : isNotApplicable
                ? "Not Applicable"
                : "Certainty"}
            </span>
          </motion.div>
        </div>

        {/* NOT_APPLICABLE explanation banner */}
        {isNotApplicable && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-6 pt-5 border-t border-white/[0.06]"
          >
            <div className="rounded-xl border border-[#a78bfa]/20 bg-[#a78bfa]/5 px-4 py-3 text-xs text-[#a78bfa]/90 font-mono leading-relaxed">
              <span className="font-bold text-[#a78bfa]">ℹ️ Why no deepfake score?</span>
              <br />
              Deepfake forensics analyze biometric human features (facial geometry, skin
              micro-texture, boundary blending). This content was identified as{" "}
              <span className="text-white font-semibold">{categoryLabel}</span> — a
              non-photorealistic media type. Running neural deepfake models on artwork
              would produce statistically meaningless scores.
            </div>
          </motion.div>
        )}

        {/* Dual-Sided Probability Slider — only for REAL / FAKE / UNCERTAIN */}
        {!isNotApplicable && (
          <div className="mt-8 pt-5 border-t border-white/[0.06]">
            <div className="flex justify-between items-center text-xs font-mono mb-3">
              <span className="text-[#10b981] font-semibold flex items-center gap-1.5">
                <ShieldCheck className="h-3.5 w-3.5" />
                Real / Authenticity: {realPct}%
              </span>
              <span className="text-[#ef4444] font-semibold flex items-center gap-1.5">
                Deepfake / Synthetic: {fakePct}%
                <ShieldAlert className="h-3.5 w-3.5" />
              </span>
            </div>

            {/* Dual bar */}
            <div className="h-3 w-full rounded-full bg-white/[0.06] overflow-hidden flex">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${realPct}%` }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                className="h-full bg-gradient-to-r from-[#10b981] to-[#38bdf8]"
              />
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${fakePct}%` }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                className="h-full bg-gradient-to-r from-[#ef4444] to-[#f59e0b]"
              />
            </div>

            {/* Center divider marker */}
            <div className="relative h-0">
              <div
                className="absolute top-[-14px] w-0.5 h-5 bg-white/30 rounded-full"
                style={{ left: "50%", transform: "translateX(-50%)" }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

