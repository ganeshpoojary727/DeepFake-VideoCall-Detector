"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Scan, Sparkles } from "lucide-react";

interface HolographicScannerProps {
  mediaType: string;
  fileName: string;
}

const SCAN_STEPS = [
  "Initializing GPU CUDA neural pipeline...",
  "Running YuNet DNN face detection & 20% margin alignment...",
  "Sampling 16 uniform spatiotemporal sequence frames...",
  "Passing through EfficientNet-B4 spatial feature encoder...",
  "Evaluating multi-head temporal self-attention transformer...",
  "Decomposing audio stream into 16kHz raw waveform...",
  "Computing AASIST spectro-temporal graph attention...",
  "Calculating weighted late cross-modal fusion...",
  "Synthesizing explainable AI forensic factor report...",
];

export default function HolographicScanner({ mediaType, fileName }: HolographicScannerProps) {
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStepIndex((prev) => (prev + 1) % SCAN_STEPS.length);
    }, 600);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative flex flex-col items-center justify-center overflow-hidden rounded-2xl border border-[#38bdf8]/20 bg-[#07080f]/90 p-12 text-center shadow-[0_0_50px_rgba(56,189,248,0.15)] backdrop-blur-2xl">
      {/* Background Cyber Grid & Glow */}
      <div className="absolute inset-0 cyber-grid opacity-30 pointer-events-none" />
      <div className="laser-line pointer-events-none" />

      {/* Pulsing Radar Ring */}
      <div className="relative mb-6 flex h-28 w-28 items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
          className="absolute inset-0 rounded-full border-2 border-dashed border-[#38bdf8]/30"
        />
        <motion.div
          animate={{ scale: [1, 1.25, 1], opacity: [0.3, 0.7, 0.3] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="absolute inset-2 rounded-full bg-[#38bdf8]/15 blur-md"
        />
        <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-[#38bdf8] to-[#6366f1] shadow-[0_0_30px_rgba(56,189,248,0.4)]">
          <Scan className="h-8 w-8 text-white animate-pulse" />
        </div>
      </div>

      {/* Target Title & File */}
      <h3 className="mb-1 text-xl font-bold tracking-tight text-white glow-cyan">
        NEURAL DEEPFAKE SCAN IN PROGRESS
      </h3>
      <p className="mb-6 font-mono text-xs text-[#38bdf8]">
        Target: <span className="text-[#9ca3af]">{fileName}</span> • Type:{" "}
        <span className="uppercase text-[#a5b4fc]">{mediaType}</span>
      </p>

      {/* Animated Step Ticker */}
      <div className="flex h-10 items-center justify-center rounded-full border border-white/[0.08] bg-black/40 px-6 py-2">
        <motion.div
          key={stepIndex}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="flex items-center gap-2 font-mono text-xs text-[#9ca3af]"
        >
          <Sparkles className="h-3.5 w-3.5 text-[#38bdf8] animate-spin" />
          <span>{SCAN_STEPS[stepIndex]}</span>
        </motion.div>
      </div>
    </div>
  );
}
