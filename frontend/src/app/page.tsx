"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Sparkles, AlertCircle, ArrowRight, UploadCloud, Video, Image as ImageIcon, Music } from "lucide-react";
import CustomCursor from "../components/CustomCursor";
import Navbar from "../components/Navbar";
import MediaUploader from "../components/MediaUploader";
import HolographicScanner from "../components/HolographicScanner";
import ForensicReportView from "../components/ForensicReportView";
import BatchScannerView from "../components/BatchScannerView";
import SystemTelemetryHUD from "../components/SystemTelemetryHUD";
import { analyzeMediaFile } from "../lib/api";
import { AnalysisReport } from "../lib/types";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"single" | "batch" | "system">("single");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setReport(null);
    setErrorMsg(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setReport(null);
    setErrorMsg(null);
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;
    setIsScanning(true);
    setErrorMsg(null);
    try {
      const res = await analyzeMediaFile(selectedFile);
      setReport(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to analyze media file.");
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <main className="relative min-h-screen bg-[#06070d] text-white selection:bg-sky-500/30 selection:text-sky-300">
      {/* High-Budget Magnetic Fluid Cursor */}
      <CustomCursor />

      {/* Ambient Lighting Mesh */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 h-[550px] w-[550px] rounded-full bg-sky-600/15 blur-[120px]" />
        <div className="absolute top-1/3 -right-40 h-[500px] w-[500px] rounded-full bg-purple-600/15 blur-[140px]" />
        <div className="absolute -bottom-40 left-1/3 h-[600px] w-[600px] rounded-full bg-indigo-600/10 blur-[150px]" />
        <div className="absolute inset-0 cyber-grid opacity-20" />
      </div>

      {/* Navigation Bar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Container */}
      <div className="relative z-10 mx-auto max-w-7xl px-6 py-10">
        <AnimatePresence mode="wait">
          {activeTab === "single" && (
            <motion.div
              key="single-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              className="space-y-10"
            >
              {/* Hero Banner */}
              {!report && !isScanning && (
                <div className="text-center max-w-3xl mx-auto space-y-4">
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="inline-flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-500/10 px-4 py-1.5 text-xs font-semibold text-sky-400 shadow-[0_0_20px_rgba(56,189,248,0.25)]"
                  >
                    <Sparkles className="h-3.5 w-3.5 text-sky-400" />
                    <span>Next-Gen Explainable AI Deepfake Forensics</span>
                  </motion.div>

                  <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight">
                    Verify Media Integrity With{" "}
                    <span className="bg-gradient-to-r from-sky-400 via-indigo-300 to-purple-400 bg-clip-text text-transparent glow-cyan">
                      Neural Precision
                    </span>
                  </h1>

                  <p className="text-sm md:text-base text-slate-400 leading-relaxed max-w-2xl mx-auto">
                    Examine photos, video streams, and audio recordings using dual-stream spatiotemporal transformers and graph attention anti-spoofing networks.
                  </p>
                </div>
              )}

              {/* Upload & Inspection Workspace */}
              <div className="mx-auto max-w-4xl space-y-6">
                {!report && !isScanning && (
                  <div className="space-y-6">
                    <MediaUploader
                      onFileSelected={handleFileSelect}
                      selectedFile={selectedFile}
                      onClear={handleClear}
                      isScanning={isScanning}
                    />

                    {selectedFile && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex justify-center"
                      >
                        <button
                          onClick={handleStartAnalysis}
                          className="flex items-center gap-3 rounded-2xl bg-gradient-to-r from-sky-500 via-indigo-600 to-purple-600 px-8 py-4 text-sm font-bold text-white shadow-[0_0_35px_rgba(56,189,248,0.5)] hover:scale-[1.02] hover:shadow-[0_0_45px_rgba(56,189,248,0.7)] transition-all duration-300"
                        >
                          <span>Execute Forensic Neural Scan</span>
                          <ArrowRight className="h-4 w-4" />
                        </button>
                      </motion.div>
                    )}
                  </div>
                )}

                {/* Error Banner */}
                {errorMsg && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center gap-3 rounded-2xl border border-red-500/30 bg-red-500/10 p-4 text-xs font-semibold text-red-300 shadow-[0_0_20px_rgba(239,68,68,0.2)]"
                  >
                    <AlertCircle className="h-5 w-5 text-red-400 shrink-0" />
                    <span>{errorMsg}</span>
                  </motion.div>
                )}

                {/* Holographic Laser Scanner */}
                {isScanning && selectedFile && (
                  <HolographicScanner
                    mediaType={selectedFile.name.split(".").pop() || "media"}
                    fileName={selectedFile.name}
                  />
                )}

                {/* Forensic Diagnostic Report */}
                {report && (
                  <ForensicReportView report={report} onReset={handleClear} />
                )}
              </div>
            </motion.div>
          )}

          {activeTab === "batch" && (
            <motion.div
              key="batch-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
            >
              <BatchScannerView />
            </motion.div>
          )}

          {activeTab === "system" && (
            <motion.div
              key="system-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
            >
              <SystemTelemetryHUD />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <footer className="relative z-10 mt-20 border-t border-white/10 py-8 text-center text-xs text-slate-500">
        <div className="mx-auto max-w-7xl flex flex-col md:flex-row items-center justify-between px-6 gap-4">
          <p>🛡️ DeepGuard AI • Multi-Modal Forensic Deepfake Detection</p>
          <p className="font-mono text-[11px] text-slate-400">
            AASIST (ASVspoof 2019) • EfficientNet-B4 + Temporal Transformer • YuNet
          </p>
        </div>
      </footer>
    </main>
  );
}
