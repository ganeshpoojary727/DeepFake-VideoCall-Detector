"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Shield,
  ArrowRight,
  Zap,
  ScanEye,
  AudioLines,
  Binary,
  AlertCircle,
} from "lucide-react";
import CustomCursor from "../components/CustomCursor";
import Navbar from "../components/Navbar";
import MediaUploader from "../components/MediaUploader";
import ForensicMetricsTicker from "../components/ForensicMetricsTicker";
import PipelineCards from "../components/PipelineCards";
import Footer from "../components/Footer";
import { analyzeMediaFile } from "../lib/api";
import { ScanSession } from "../lib/types";

export default function Home() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setErrorMsg(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setErrorMsg(null);
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;
    setIsScanning(true);
    setErrorMsg(null);

    try {
      const report = await analyzeMediaFile(selectedFile);

      // Generate scan ID
      const scanId = `DF-${Date.now().toString(36).toUpperCase()}${Math.random().toString(36).substring(2, 6).toUpperCase()}`;

      // Store in sessionStorage for the report page
      const session: ScanSession = {
        scanId,
        fileName: selectedFile.name,
        fileSize: selectedFile.size,
        mediaType: report.media_type,
        timestamp: new Date().toISOString(),
        report,
      };
      sessionStorage.setItem(`scan_${scanId}`, JSON.stringify(session));

      // Also store the file in sessionStorage as base64 for media preview
      const reader = new FileReader();
      reader.onload = () => {
        sessionStorage.setItem(`scan_file_${scanId}`, JSON.stringify({
          name: selectedFile.name,
          type: selectedFile.type,
          data: reader.result,
        }));
        // Navigate to report page
        router.push(`/report/${scanId}`);
      };
      reader.readAsDataURL(selectedFile);
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Failed to analyze media file.";
      setErrorMsg(msg);
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <main className="relative min-h-screen bg-[#07080f] text-[#e8eaed]">
      {/* Custom Cursor */}
      <CustomCursor />

      {/* Ambient Grid Background */}
      <div className="pointer-events-none fixed inset-0 z-0 cyber-grid opacity-40" />

      {/* Navbar */}
      <Navbar />

      {/* ═══════════════════════════════════════════════════════════
          HERO SECTION
          ═══════════════════════════════════════════════════════════ */}
      <section className="relative z-10 pt-20 pb-12 px-6">
        <div className="mx-auto max-w-4xl text-center space-y-6">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="inline-flex items-center gap-2 rounded-full border border-[#38bdf8]/25 bg-[#38bdf8]/8 px-4 py-1.5 text-xs font-semibold text-[#38bdf8]"
          >
            <Shield className="h-3.5 w-3.5" />
            Enterprise Deepfake Detection & Forensic Intelligence
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-4xl md:text-6xl font-extrabold tracking-tight leading-[1.1]"
          >
            Detect synthetic media with{" "}
            <span className="gradient-text-cyber">
              forensic precision
            </span>
          </motion.h1>

          {/* Subtext */}
          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="text-base md:text-lg text-[#9ca3af] max-w-2xl mx-auto leading-relaxed"
          >
            Upload images, videos, or audio files to analyze for GAN-generated
            artifacts, neural vocoder footprints, and spatiotemporal anomalies
            with explainable AI diagnostics.
          </motion.p>

          {/* Quick Stats Row */}
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            id="capabilities"
            className="flex flex-wrap items-center justify-center gap-4 pt-2"
          >
            {[
              { icon: AudioLines, label: "AASIST EER 0.52%", color: "text-[#38bdf8]" },
              { icon: ScanEye, label: "97.3% Video AUC", color: "text-[#6366f1]" },
              { icon: Binary, label: "FFT + ELA + SRM", color: "text-[#10b981]" },
              { icon: Zap, label: "CUDA Accelerated", color: "text-[#f59e0b]" },
            ].map((stat) => (
              <div
                key={stat.label}
                className="flex items-center gap-1.5 text-xs font-mono text-[#6b7280]"
              >
                <stat.icon className={`h-3.5 w-3.5 ${stat.color}`} />
                <span>{stat.label}</span>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          SCANNER / UPLOAD SANDBOX
          ═══════════════════════════════════════════════════════════ */}
      <section className="relative z-10 px-6 pb-10">
        <div className="mx-auto max-w-2xl">
          {/* Error Banner */}
          {errorMsg && (
            <motion.div
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-3 rounded-xl border border-[#ef4444]/30 bg-[#ef4444]/8 p-4 text-xs font-semibold text-[#ef4444] mb-6"
            >
              <AlertCircle className="h-4.5 w-4.5 shrink-0" />
              <span>{errorMsg}</span>
            </motion.div>
          )}

          <MediaUploader
            onFileSelected={handleFileSelect}
            selectedFile={selectedFile}
            onClear={handleClear}
            isScanning={isScanning}
            onStartAnalysis={handleStartAnalysis}
          />
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          METRICS TICKER
          ═══════════════════════════════════════════════════════════ */}
      <div id="benchmarks">
        <ForensicMetricsTicker />
      </div>

      {/* ═══════════════════════════════════════════════════════════
          PIPELINE "UNDER THE HOOD" SECTION
          ═══════════════════════════════════════════════════════════ */}
      <PipelineCards />

      {/* ═══════════════════════════════════════════════════════════
          CTA SECTION
          ═══════════════════════════════════════════════════════════ */}
      <section className="relative z-10 py-20 px-6">
        <div className="mx-auto max-w-3xl text-center">
          <div className="glass-card p-10 md:p-14 space-y-6 border border-[#38bdf8]/10">
            {/* Ambient glow */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-[#38bdf8]/5 via-transparent to-[#6366f1]/5 pointer-events-none" />

            <h2 className="text-2xl md:text-3xl font-extrabold text-white relative">
              Ready to analyze your media?
            </h2>
            <p className="text-sm text-[#9ca3af] relative">
              Start a forensic scan to receive a comprehensive deepfake analysis
              report with explainable evidence factors.
            </p>
            <button
              onClick={() => {
                const el = document.getElementById("scanner");
                if (el) el.scrollIntoView({ behavior: "smooth" });
              }}
              className="relative inline-flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-[#38bdf8] to-[#6366f1] px-8 py-3.5 text-sm font-bold text-white shadow-[0_0_30px_rgba(56,189,248,0.25)] hover:shadow-[0_0_40px_rgba(56,189,248,0.40)] transition-all hover:scale-[1.02]"
            >
              <Shield className="h-4 w-4" />
              Launch Forensic Scan
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          FOOTER
          ═══════════════════════════════════════════════════════════ */}
      <Footer />
    </main>
  );
}
