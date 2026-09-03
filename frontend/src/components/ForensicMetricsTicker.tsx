"use client";

import { motion } from "framer-motion";
import {
  Zap,
  AudioLines,
  ScanEye,
  Binary,
  Layers,
} from "lucide-react";

const METRICS = [
  { icon: AudioLines, label: "AASIST Audio", value: "EER 0.52%", accent: "text-[#38bdf8]" },
  { icon: ScanEye, label: "EfficientNet-B4", value: "Spatiotemporal Transformer", accent: "text-[#6366f1]" },
  { icon: Zap, label: "YuNet ONNX", value: "Facial Alignment", accent: "text-[#10b981]" },
  { icon: Binary, label: "2D FFT", value: "Fourier Forensics", accent: "text-[#f59e0b]" },
  { icon: Layers, label: "Multimodal Fusion", value: "0.6A + 0.4V Calibrated", accent: "text-[#38bdf8]" },
  { icon: AudioLines, label: "ASVspoof5 Protocol", value: "99.71% Accuracy", accent: "text-[#10b981]" },
  { icon: ScanEye, label: "FF++ Benchmark", value: "97.3% AUC", accent: "text-[#6366f1]" },
  { icon: Binary, label: "SRM + ELA", value: "Sensor Noise Analysis", accent: "text-[#f59e0b]" },
];

export default function ForensicMetricsTicker() {
  return (
    <div className="relative w-full py-4 border-y border-white/[0.04] bg-[#07080f]/80 backdrop-blur-sm">
      <div className="ticker-container">
        <motion.div
          className="flex gap-8 whitespace-nowrap animate-ticker-scroll"
          style={{ width: "max-content" }}
        >
          {/* Duplicate items for seamless loop */}
          {[...METRICS, ...METRICS].map((metric, idx) => {
            const Icon = metric.icon;
            return (
              <div
                key={idx}
                className="flex items-center gap-2.5 px-4 py-1.5 rounded-full border border-white/[0.06] bg-white/[0.02]"
              >
                <Icon className={`h-3.5 w-3.5 ${metric.accent} shrink-0`} />
                <span className="text-xs font-semibold text-white">
                  {metric.label}
                </span>
                <span className="text-[11px] font-mono text-[#9ca3af]">
                  {metric.value}
                </span>
              </div>
            );
          })}
        </motion.div>
      </div>
    </div>
  );
}
