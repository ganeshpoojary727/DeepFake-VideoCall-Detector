"use client";

import { useState, useRef, useMemo } from "react";
import { motion } from "framer-motion";
import {
  Eye,
  Video,
  Volume2,
  Image as ImageIcon,
  Flame,
  Clock,
  Activity,
  AlertCircle,
} from "lucide-react";
import { ConsolidatedForensicReport } from "../lib/types";

interface MediaInspectorProps {
  report: ConsolidatedForensicReport;
  file?: File | null;
}

type CanvasView = "original" | "ela" | "fft";

export default function MediaInspector({ report, file }: MediaInspectorProps) {
  const [canvasView, setCanvasView] = useState<CanvasView>("original");
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);

  const previewUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  const handleSeek = (seconds: number) => {
    if (mediaRef.current) {
      mediaRef.current.currentTime = seconds;
      mediaRef.current.play().catch(() => {});
    }
  };

  const isVideo =
    report.media_type === "VIDEO" || report.media_type === "MULTIMODAL";
  const isImage = report.media_type === "IMAGE";
  const isAudio = report.media_type === "AUDIO";

  return (
    <div className="glass-card p-5 space-y-4">
      {/* Header & View Toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Eye className="h-4.5 w-4.5 text-[#38bdf8]" />
          <h3 className="text-sm font-bold text-white">
            Interactive Media Inspector
          </h3>
        </div>

        {/* View Mode Toggle */}
        {(isImage || isVideo) && (
          <div className="flex items-center rounded-xl bg-white/[0.04] p-1 border border-white/[0.06] text-xs">
            <button
              onClick={() => setCanvasView("original")}
              className={`px-3 py-1 rounded-lg font-medium transition ${
                canvasView === "original"
                  ? "bg-white/[0.08] text-white shadow-sm"
                  : "text-[#6b7280] hover:text-white"
              }`}
            >
              Original
            </button>
            <button
              onClick={() => setCanvasView("ela")}
              className={`px-3 py-1 rounded-lg font-medium flex items-center gap-1.5 transition ${
                canvasView === "ela"
                  ? "bg-[#ef4444]/10 text-[#ef4444] shadow-sm"
                  : "text-[#6b7280] hover:text-white"
              }`}
            >
              <Flame className="h-3 w-3" /> ELA
            </button>
            <button
              onClick={() => setCanvasView("fft")}
              className={`px-3 py-1 rounded-lg font-medium flex items-center gap-1.5 transition ${
                canvasView === "fft"
                  ? "bg-[#6366f1]/10 text-[#a5b4fc] shadow-sm"
                  : "text-[#6b7280] hover:text-white"
              }`}
            >
              <Activity className="h-3 w-3" /> FFT
            </button>
          </div>
        )}
      </div>

      {/* Media Canvas */}
      <div className="relative rounded-2xl bg-black/60 border border-white/[0.06] overflow-hidden flex items-center justify-center min-h-[260px] max-h-[400px]">
        {/* Image Viewer */}
        {previewUrl && isImage && (
          <div className="relative w-full h-full flex items-center justify-center p-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Forensic candidate"
              className="max-h-[380px] w-auto object-contain rounded-xl"
            />
            {canvasView === "ela" && (
              <div className="absolute inset-0 bg-gradient-to-tr from-[#ef4444]/20 via-transparent to-[#f59e0b]/20 pointer-events-none rounded-xl flex items-center justify-center">
                <span className="bg-[#0c0d14]/90 border border-[#ef4444]/30 text-[#ef4444] text-xs px-3 py-1.5 rounded-full font-mono">
                  Error Level Analysis Overlay
                </span>
              </div>
            )}
            {canvasView === "fft" && (
              <div className="absolute inset-0 bg-gradient-to-tr from-[#6366f1]/20 via-transparent to-[#38bdf8]/20 pointer-events-none rounded-xl flex items-center justify-center">
                <span className="bg-[#0c0d14]/90 border border-[#6366f1]/30 text-[#a5b4fc] text-xs px-3 py-1.5 rounded-full font-mono">
                  Fourier Power Spectrum View
                </span>
              </div>
            )}
          </div>
        )}

        {/* Video Viewer */}
        {previewUrl && isVideo && (
          <div className="relative w-full">
            <video
              ref={mediaRef as React.RefObject<HTMLVideoElement>}
              src={previewUrl}
              controls
              className="max-h-[380px] w-full rounded-xl"
            />
            {canvasView !== "original" && (
              <div className="absolute top-3 left-3 bg-[#0c0d14]/90 border border-[#6366f1]/30 text-[#a5b4fc] text-xs px-2.5 py-1 rounded-full font-mono flex items-center gap-1.5">
                <Flame className="h-3 w-3" />
                {canvasView === "ela"
                  ? "ELA Active"
                  : "FFT Spectrum Active"}
              </div>
            )}
          </div>
        )}

        {/* Audio Viewer */}
        {previewUrl && isAudio && (
          <div className="p-8 w-full flex flex-col items-center gap-4">
            <div className="h-16 w-16 rounded-2xl bg-[#6366f1]/15 border border-[#6366f1]/30 flex items-center justify-center text-[#a5b4fc]">
              <Volume2 className="h-8 w-8" />
            </div>
            <audio
              ref={mediaRef as React.RefObject<HTMLAudioElement>}
              src={previewUrl}
              controls
              className="w-full max-w-md"
            />

            {/* Audio Spoof Segment Flags */}
            {report.modality_breakdown?.audio?.timeline &&
              report.modality_breakdown.audio.timeline.length > 0 && (
                <div className="w-full max-w-md space-y-1.5 mt-2">
                  <span className="text-[11px] font-mono text-[#6b7280]">
                    Flagged acoustic segments:
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {report.modality_breakdown.audio.timeline
                      .filter(
                        (chunk) =>
                          chunk.spoof_prob !== undefined &&
                          chunk.spoof_prob > 0.5
                      )
                      .map((chunk, i) => (
                        <button
                          key={i}
                          onClick={() =>
                            handleSeek(chunk.start_time_sec ?? 0)
                          }
                          className="flex items-center gap-1 px-2 py-1 rounded-lg border border-[#ef4444]/30 bg-[#ef4444]/10 text-[#ef4444] text-[10px] font-mono hover:bg-[#ef4444]/20 transition"
                        >
                          <AlertCircle className="h-2.5 w-2.5" />
                          {(chunk.start_time_sec ?? 0).toFixed(1)}s–
                          {(chunk.end_time_sec ?? 0).toFixed(1)}s
                          <span className="font-bold">
                            ({((chunk.spoof_prob ?? 0) * 100).toFixed(0)}%)
                          </span>
                        </button>
                      ))}
                  </div>
                </div>
              )}
          </div>
        )}

        {/* No Preview Fallback */}
        {!previewUrl && (
          <div className="p-10 text-center text-[#6b7280]">
            <div className="flex items-center justify-center gap-2 mb-2">
              {isVideo && <Video className="h-5 w-5" />}
              {isImage && <ImageIcon className="h-5 w-5" />}
              {isAudio && <Volume2 className="h-5 w-5" />}
            </div>
            <p className="text-xs">
              Media preview unavailable. Forensic telemetry rendered from scan
              data.
            </p>
          </div>
        )}
      </div>

      {/* Temporal Sync Timeline */}
      {report.temporal_sync && report.temporal_sync.length > 0 && (
        <div className="pt-4 border-t border-white/[0.06] space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-white flex items-center gap-1.5">
              <Activity className="h-3.5 w-3.5 text-[#38bdf8]" />
              Temporal Anomaly Timeline
            </span>
            <span className="text-[#6b7280] font-mono text-[11px]">
              Click to seek
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {report.temporal_sync.map((sync, sIdx) => {
              const isAnomaly = sync.is_anomaly;
              return (
                <motion.button
                  key={sIdx}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: sIdx * 0.03 }}
                  onClick={() => handleSeek(sync.second)}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl border text-xs font-mono transition ${
                    isAnomaly
                      ? "border-[#ef4444]/30 bg-[#ef4444]/10 text-[#ef4444] hover:bg-[#ef4444]/20"
                      : "border-white/[0.06] bg-white/[0.02] text-[#9ca3af] hover:bg-white/[0.05]"
                  }`}
                >
                  <Clock className="h-3 w-3" />
                  <span>{sync.second.toFixed(1)}s</span>
                  <span className="text-[10px] font-bold">
                    ({(sync.fused_spoof_prob * 100).toFixed(0)}%)
                  </span>
                </motion.button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
