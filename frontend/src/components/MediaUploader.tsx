"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Image as ImageIcon,
  Video,
  Music,
  X,
  Scan,
  Upload,
  Loader2,
  Sparkles,
  FileAudio,
  FileVideo,
  FileImage,
} from "lucide-react";

interface MediaUploaderProps {
  onFileSelected: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  isScanning: boolean;
  onStartAnalysis?: () => void;
}

type ModalityTab = "image" | "video" | "audio";

const SUPPORTED_EXTS: Record<ModalityTab, string[]> = {
  image: ["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
  video: ["mp4", "avi", "mov", "mkv", "webm"],
  audio: ["wav", "mp3", "flac", "ogg", "m4a"],
};

const ALL_EXTS = Object.values(SUPPORTED_EXTS).flat();

const SCANNING_PHASES = [
  "Demuxing media streams & validating container integrity...",
  "Extracting facial embeddings via YuNet ONNX alignment...",
  "Computing AASIST spectro-temporal graph attention features...",
  "Evaluating 2D FFT periodic lattice & ELA residuals...",
  "Running EfficientNet-B4 spatiotemporal transformer...",
  "Fusing multimodal telemetry & synthesizing forensic report...",
];

const MODALITY_TABS: { id: ModalityTab; label: string; icon: React.ElementType; emoji: string }[] = [
  { id: "image", label: "Image", icon: ImageIcon, emoji: "🖼️" },
  { id: "video", label: "Video", icon: Video, emoji: "🎥" },
  { id: "audio", label: "Audio", icon: Music, emoji: "🎙️" },
];

export default function MediaUploader({
  onFileSelected,
  selectedFile,
  onClear,
  isScanning,
  onStartAnalysis,
}: MediaUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [activeTab, setActiveTab] = useState<ModalityTab>("image");
  const [phaseIdx, setPhaseIdx] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isScanning) return;
    setPhaseIdx(0);
    const interval = setInterval(() => {
      setPhaseIdx((prev) => (prev + 1) % SCANNING_PHASES.length);
    }, 2200);
    return () => clearInterval(interval);
  }, [isScanning]);

  const handleFile = (file: File) => {
    onFileSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const getMediaIcon = (name: string) => {
    const ext = name.split(".").pop()?.toLowerCase() || "";
    if (SUPPORTED_EXTS.image.includes(ext)) return FileImage;
    if (SUPPORTED_EXTS.video.includes(ext)) return FileVideo;
    if (SUPPORTED_EXTS.audio.includes(ext)) return FileAudio;
    return FileImage;
  };

  return (
    <div id="scanner" className="w-full space-y-5">
      {/* ── Scanning Progress Overlay ── */}
      <AnimatePresence>
        {isScanning && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="glass-card p-6 border border-[#38bdf8]/20 box-glow-cyan relative overflow-hidden"
          >
            {/* Laser scan line */}
            <div className="laser-line" />

            <div className="relative z-10 flex items-center gap-4">
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-[#38bdf8]/10 border border-[#38bdf8]/30 text-[#38bdf8]">
                <Loader2 className="h-7 w-7 animate-spin" />
              </div>
              <div className="space-y-1.5 flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[#38bdf8] uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles className="h-3.5 w-3.5" />
                    Neural Forensic Scan in Progress
                  </span>
                  <span className="text-xs font-mono text-[#6b7280]">
                    Phase {phaseIdx + 1}/{SCANNING_PHASES.length}
                  </span>
                </div>
                <AnimatePresence mode="wait">
                  <motion.p
                    key={phaseIdx}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    className="text-sm font-medium text-white truncate"
                  >
                    {SCANNING_PHASES[phaseIdx]}
                  </motion.p>
                </AnimatePresence>
              </div>
            </div>

            {/* Shimmer progress bar */}
            <div className="mt-4 h-1.5 w-full rounded-full bg-white/[0.06] overflow-hidden">
              <div className="h-full w-full animate-shimmer rounded-full" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Modality Tabs ── */}
      {!isScanning && (
        <div className="flex items-center justify-center gap-2">
          {MODALITY_TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all border ${
                activeTab === tab.id
                  ? "border-[#38bdf8]/30 bg-[#38bdf8]/10 text-[#38bdf8] shadow-[0_0_15px_rgba(56,189,248,0.15)]"
                  : "border-white/[0.06] bg-white/[0.02] text-[#6b7280] hover:text-white hover:border-white/[0.12]"
              }`}
            >
              <span>{tab.emoji}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      )}

      {/* ── Dropzone ── */}
      {!isScanning && (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`dropzone-border rounded-2xl p-8 transition-all duration-300 cursor-pointer ${
            isDragging
              ? "active bg-[#38bdf8]/[0.03]"
              : "hover:bg-white/[0.01]"
          }`}
          onClick={() => !selectedFile && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={SUPPORTED_EXTS[activeTab]
              .map((e) => `.${e}`)
              .join(",")}
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.[0]) handleFile(e.target.files[0]);
            }}
          />

          {selectedFile ? (
            /* ── File Selected State ── */
            <div className="flex flex-col items-center gap-4">
              <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-white/[0.03] border border-white/[0.08] w-full max-w-lg">
                {(() => {
                  const Icon = getMediaIcon(selectedFile.name);
                  return (
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#38bdf8]/10 border border-[#38bdf8]/20 text-[#38bdf8]">
                      <Icon className="h-5 w-5" />
                    </div>
                  );
                })()}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-white truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-[11px] font-mono text-[#6b7280]">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onClear();
                  }}
                  className="p-1.5 rounded-lg text-[#6b7280] hover:text-[#ef4444] hover:bg-[#ef4444]/10 transition"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {/* Launch Scan Button */}
              {onStartAnalysis && (
                <motion.button
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onStartAnalysis();
                  }}
                  className="flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-[#38bdf8] to-[#6366f1] px-8 py-3 text-sm font-bold text-white shadow-[0_0_30px_rgba(56,189,248,0.25)] hover:shadow-[0_0_40px_rgba(56,189,248,0.40)] transition-all hover:scale-[1.02]"
                >
                  <Scan className="h-4 w-4" />
                  Launch Forensic Scan
                </motion.button>
              )}
            </div>
          ) : (
            /* ── Empty Dropzone State ── */
            <div className="flex flex-col items-center gap-4 text-center">
              <motion.div
                animate={{ y: [0, -4, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                className="flex h-16 w-16 items-center justify-center rounded-2xl bg-[#38bdf8]/10 border border-[#38bdf8]/20 text-[#38bdf8]"
              >
                <Upload className="h-7 w-7" />
              </motion.div>
              <div>
                <p className="text-sm font-semibold text-white mb-1">
                  Drop your {activeTab} file here
                </p>
                <p className="text-xs text-[#6b7280]">
                  or click to browse •{" "}
                  <span className="font-mono text-[#38bdf8]/70">
                    {SUPPORTED_EXTS[activeTab]
                      .map((e) => `.${e}`)
                      .join(", ")}
                  </span>
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
