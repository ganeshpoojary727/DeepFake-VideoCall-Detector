"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, Image, Video, Music, FileCheck, X, Sparkles } from "lucide-react";

interface MediaUploaderProps {
  onFileSelected: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  isScanning: boolean;
}

const SUPPORTED_EXTS = [
  "jpg", "jpeg", "png", "webp", "bmp", "tiff",
  "mp4", "avi", "mov", "mkv", "webm",
  "wav", "mp3", "flac", "ogg", "m4a",
];

export default function MediaUploader({
  onFileSelected,
  selectedFile,
  onClear,
  isScanning,
}: MediaUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    onFileSelected(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const getMediaType = (name: string) => {
    const ext = name.split(".").pop()?.toLowerCase();
    if (["jpg", "jpeg", "png", "webp", "bmp", "tiff"].includes(ext || "")) return "image";
    if (["mp4", "avi", "mov", "mkv", "webm"].includes(ext || "")) return "video";
    if (["wav", "mp3", "flac", "ogg", "m4a"].includes(ext || "")) return "audio";
    return "unknown";
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept={SUPPORTED_EXTS.map((e) => `.${e}`).join(",")}
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
          }
        }}
      />

      <AnimatePresence mode="wait">
        {!selectedFile ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={`glass-panel-interactive relative flex flex-col items-center justify-center rounded-3xl p-12 text-center cursor-pointer transition-all duration-300 ${
              isDragging
                ? "border-sky-400 bg-sky-500/10 shadow-[0_0_40px_rgba(56,189,248,0.3)] scale-[1.01]"
                : "border-white/10 hover:border-sky-500/30"
            }`}
          >
            {/* Glowing Upload Orb */}
            <div className="relative mb-6 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-tr from-sky-500/20 to-indigo-500/20 border border-sky-400/30 shadow-[0_0_25px_rgba(56,189,248,0.2)]">
              <UploadCloud className="h-10 w-10 text-sky-400 animate-bounce" />
            </div>

            <h3 className="mb-2 text-xl font-bold text-white">
              Drag & Drop candidate media file here
            </h3>
            <p className="mb-6 text-sm text-slate-400 max-w-md">
              Supports photos, video clips, and voice recordings up to 500 MB for neural forensic verification.
            </p>

            {/* Supported Badges */}
            <div className="flex flex-wrap justify-center gap-2">
              <span className="flex items-center gap-1.5 rounded-full border border-sky-500/20 bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-300">
                <Image className="h-3.5 w-3.5" /> Images (.jpg, .png, .webp)
              </span>
              <span className="flex items-center gap-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-3 py-1 text-xs font-medium text-indigo-300">
                <Video className="h-3.5 w-3.5" /> Videos (.mp4, .mov, .avi)
              </span>
              <span className="flex items-center gap-1.5 rounded-full border border-purple-500/20 bg-purple-500/10 px-3 py-1 text-xs font-medium text-purple-300">
                <Music className="h-3.5 w-3.5" /> Audio (.wav, .mp3, .flac)
              </span>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="glass-panel relative overflow-hidden rounded-3xl p-6 border border-white/10"
          >
            {/* Header / Info Bar */}
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-500/10 border border-sky-500/30 text-sky-400">
                  <FileCheck className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="font-semibold text-white truncate max-w-sm">
                    {selectedFile.name}
                  </h4>
                  <p className="text-xs text-slate-400 font-mono">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {getMediaType(selectedFile.name).toUpperCase()}
                  </p>
                </div>
              </div>

              {!isScanning && (
                <button
                  onClick={onClear}
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white transition"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            {/* Media Player / Viewer */}
            <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/60 flex items-center justify-center min-h-[260px] max-h-[380px]">
              {previewUrl && getMediaType(selectedFile.name) === "image" && (
                <img
                  src={previewUrl}
                  alt="Upload preview"
                  className="max-h-[360px] w-auto object-contain rounded-xl"
                />
              )}

              {previewUrl && getMediaType(selectedFile.name) === "video" && (
                <video
                  src={previewUrl}
                  controls
                  className="max-h-[360px] w-full rounded-xl"
                />
              )}

              {previewUrl && getMediaType(selectedFile.name) === "audio" && (
                <div className="p-8 w-full flex flex-col items-center gap-4">
                  <div className="h-16 w-16 rounded-full bg-purple-500/20 border border-purple-400/40 flex items-center justify-center text-purple-400 shadow-[0_0_20px_rgba(168,85,247,0.3)]">
                    <Music className="h-8 w-8" />
                  </div>
                  <audio src={previewUrl} controls className="w-full max-w-md" />
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
