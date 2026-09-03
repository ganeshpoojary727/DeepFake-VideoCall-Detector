"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AudioLines,
  Video,
  Image,
  Combine,
  ChevronRight,
  Waves,
  Scan,
  Binary,
  GitMerge,
} from "lucide-react";

interface PipelineCard {
  id: string;
  title: string;
  icon: React.ElementType;
  accentColor: string;
  accentBg: string;
  borderColor: string;
  glowClass: string;
  features: { name: string; description: string }[];
}

const PIPELINE_CARDS: PipelineCard[] = [
  {
    id: "audio",
    title: "Audio Subsystem",
    icon: AudioLines,
    accentColor: "text-[#38bdf8]",
    accentBg: "bg-[#38bdf8]/10",
    borderColor: "border-[#38bdf8]/20",
    glowClass: "box-glow-cyan",
    features: [
      {
        name: "AASIST Graph Attention Network",
        description: "Heterogeneous spectro-temporal graph attention on raw 1D waveform — no hand-crafted features.",
      },
      {
        name: "Mel-Spectrogram & STFT Phase",
        description: "Detects phase discontinuities and spectral rolloff artifacts left by neural vocoders.",
      },
      {
        name: "Vocoder Footprint Extraction",
        description: "Identifies characteristic GAN generator patterns (WaveGlow, HiFi-GAN, VITS) via residual analysis.",
      },
    ],
  },
  {
    id: "video",
    title: "Video Subsystem",
    icon: Video,
    accentColor: "text-[#6366f1]",
    accentBg: "bg-[#6366f1]/10",
    borderColor: "border-[#6366f1]/20",
    glowClass: "box-glow-indigo",
    features: [
      {
        name: "16-Frame Spatiotemporal Modeling",
        description: "Uniform temporal sampling with multi-head self-attention transformer over EfficientNet-B4 spatial features.",
      },
      {
        name: "Inter-Frame Blend Boundary Analysis",
        description: "Detects blending seams and warping artifacts at face swap boundaries across consecutive frames.",
      },
      {
        name: "Facial Landmark Kinematics",
        description: "YuNet ONNX face detection with 20% margin, tracking 68-point landmark temporal consistency.",
      },
    ],
  },
  {
    id: "image",
    title: "Image Subsystem",
    icon: Image,
    accentColor: "text-[#10b981]",
    accentBg: "bg-[#10b981]/10",
    borderColor: "border-[#10b981]/20",
    glowClass: "box-glow-emerald",
    features: [
      {
        name: "2D FFT Fourier Power Spectrum",
        description: "Reveals periodic lattice patterns (checkerboard artifacts) from GAN upsampling convolutions.",
      },
      {
        name: "SRM Sensor Noise Residuals (PRNU)",
        description: "Spatial Rich Model high-pass filters extract camera sensor fingerprint inconsistencies.",
      },
      {
        name: "Error Level Analysis (ELA)",
        description: "JPEG re-compression delta analysis exposing spliced or AI-generated regions.",
      },
    ],
  },
  {
    id: "fusion",
    title: "Multimodal Fusion",
    icon: Combine,
    accentColor: "text-[#f59e0b]",
    accentBg: "bg-[#f59e0b]/10",
    borderColor: "border-[#f59e0b]/20",
    glowClass: "box-glow-crimson",
    features: [
      {
        name: "Calibrated Late Fusion",
        description: "Weighted probability fusion: P = 0.6 × Audio + 0.4 × Video with temperature scaling calibration.",
      },
      {
        name: "Cross-Modal Anomaly Detection",
        description: "Detects temporal audio-visual desynchronization and lip-sync inconsistencies.",
      },
      {
        name: "Explainable AI Diagnostics",
        description: "Generates forensic factor breakdown with natural-language narrative for each detection dimension.",
      },
    ],
  },
];

const SUB_ICONS: Record<string, React.ElementType> = {
  audio: Waves,
  video: Scan,
  image: Binary,
  fusion: GitMerge,
};

export default function PipelineCards() {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <section id="pipeline" className="py-20 px-6">
      <div className="mx-auto max-w-7xl">
        {/* Section Header */}
        <div className="text-center mb-14 space-y-3">
          <span className="inline-flex items-center gap-2 rounded-full border border-[#6366f1]/30 bg-[#6366f1]/10 px-3.5 py-1 text-xs font-semibold text-[#a5b4fc]">
            <Binary className="h-3.5 w-3.5" />
            Under The Hood
          </span>
          <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white">
            Forensic Detection{" "}
            <span className="gradient-text-cyber">Pipeline</span>
          </h2>
          <p className="text-sm text-[#9ca3af] max-w-2xl mx-auto">
            Four specialized neural subsystems working in concert to analyze every dimension of media authenticity.
          </p>
        </div>

        {/* Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {PIPELINE_CARDS.map((card, cardIdx) => {
            const Icon = card.icon;
            const SubIcon = SUB_ICONS[card.id];
            const isExpanded = expandedId === card.id;

            return (
              <motion.div
                key={card.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ delay: cardIdx * 0.1, duration: 0.5 }}
                className={`glass-card p-6 cursor-pointer transition-all duration-300 ${
                  isExpanded ? `${card.borderColor} ${card.glowClass}` : ""
                }`}
                onClick={() =>
                  setExpandedId(isExpanded ? null : card.id)
                }
              >
                {/* Card Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div
                      className={`flex h-10 w-10 items-center justify-center rounded-xl ${card.accentBg} ${card.borderColor} border`}
                    >
                      <Icon className={`h-5 w-5 ${card.accentColor}`} />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        {card.title}
                      </h3>
                      <p className="text-[11px] text-[#6b7280] font-mono">
                        {card.features.length} detection modules
                      </p>
                    </div>
                  </div>
                  <motion.div
                    animate={{ rotate: isExpanded ? 90 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronRight
                      className={`h-4 w-4 ${card.accentColor} opacity-60`}
                    />
                  </motion.div>
                </div>

                {/* Expanded Features */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                      className="overflow-hidden"
                    >
                      <div className="space-y-3 pt-3 border-t border-white/[0.06]">
                        {card.features.map((feature, fIdx) => (
                          <div
                            key={fIdx}
                            className="flex items-start gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/[0.04]"
                          >
                            <SubIcon
                              className={`h-4 w-4 ${card.accentColor} mt-0.5 shrink-0`}
                            />
                            <div>
                              <h4 className="text-xs font-bold text-white mb-0.5">
                                {feature.name}
                              </h4>
                              <p className="text-[11px] text-[#9ca3af] leading-relaxed">
                                {feature.description}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Collapsed Preview */}
                {!isExpanded && (
                  <div className="flex flex-wrap gap-2 mt-1">
                    {card.features.map((feature, fIdx) => (
                      <span
                        key={fIdx}
                        className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${card.accentBg} ${card.accentColor} border ${card.borderColor}`}
                      >
                        {feature.name.split(" ").slice(0, 3).join(" ")}
                      </span>
                    ))}
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
