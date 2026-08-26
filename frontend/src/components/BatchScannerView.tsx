"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud, FileText, CheckCircle2, AlertTriangle, ShieldAlert, Download, RefreshCw } from "lucide-react";
import { analyzeMediaBatch } from "../lib/api";
import { AnalysisReport } from "../lib/types";

export default function BatchScannerView() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [reports, setReports] = useState<AnalysisReport[]>([]);
  const [filter, setFilter] = useState<"ALL" | "FAKE" | "REAL" | "UNCERTAIN">("ALL");

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      setSelectedFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleStartBatch = async () => {
    if (selectedFiles.length === 0) return;
    setIsProcessing(true);
    try {
      const res = await analyzeMediaBatch(selectedFiles);
      setReports(res);
    } catch (err: any) {
      alert(`Batch processing error: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadCsv = () => {
    if (reports.length === 0) return;
    const headers = ["File Name", "Verdict", "Confidence (%)", "Media Type", "Processing Time (ms)"];
    const rows = reports.map((r) => [
      r.metadata.original_filename || r.metadata.file_name || "unknown",
      r.verdict,
      (r.confidence * 100).toFixed(2),
      r.media_type,
      r.processing_time_ms.toFixed(1),
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "deepfake_batch_audit.csv");
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const filteredReports = reports.filter((r) => filter === "ALL" || r.verdict === filter);

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className="glass-panel-interactive rounded-3xl p-10 text-center border border-white/10"
      >
        <div className="mb-4 flex h-16 w-16 mx-auto items-center justify-center rounded-2xl bg-indigo-500/20 text-indigo-400 border border-indigo-400/30">
          <UploadCloud className="h-8 w-8" />
        </div>
        <h3 className="text-xl font-bold text-white mb-1">Batch Media Verification Hub</h3>
        <p className="text-xs text-slate-400 mb-6 max-w-md mx-auto">
          Upload collections of Images, Videos, and Audio files to execute concurrent neural deepfake forensics.
        </p>

        <input
          type="file"
          multiple
          id="batch-input"
          className="hidden"
          onChange={(e) => {
            if (e.target.files) setSelectedFiles(Array.from(e.target.files));
          }}
        />

        <div className="flex items-center justify-center gap-4">
          <label
            htmlFor="batch-input"
            className="cursor-pointer rounded-xl bg-white/10 px-5 py-2.5 text-xs font-bold text-white hover:bg-white/20 transition border border-white/10"
          >
            Choose Multiple Files
          </label>
          {selectedFiles.length > 0 && (
            <button
              onClick={handleStartBatch}
              disabled={isProcessing}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 px-6 py-2.5 text-xs font-bold text-white shadow-[0_0_20px_rgba(56,189,248,0.4)] hover:opacity-90 transition disabled:opacity-50"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" /> Processing {selectedFiles.length} files...
                </>
              ) : (
                `Process ${selectedFiles.length} File(s)`
              )}
            </button>
          )}
        </div>
      </div>

      {/* Results Dashboard */}
      {reports.length > 0 && (
        <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          {/* Summary Counters */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="glass-panel p-5 rounded-2xl border border-white/10 text-center">
              <span className="text-xs text-slate-400">Total Analyzed</span>
              <p className="text-3xl font-extrabold text-white mt-1">{reports.length}</p>
            </div>
            <div className="glass-panel p-5 rounded-2xl border border-red-500/30 bg-red-950/20 text-center">
              <span className="text-xs text-red-400">Deepfakes Identified</span>
              <p className="text-3xl font-extrabold text-red-400 mt-1">
                {reports.filter((r) => r.verdict === "FAKE").length}
              </p>
            </div>
            <div className="glass-panel p-5 rounded-2xl border border-emerald-500/30 bg-emerald-950/20 text-center">
              <span className="text-xs text-emerald-400">Authentic Media</span>
              <p className="text-3xl font-extrabold text-emerald-400 mt-1">
                {reports.filter((r) => r.verdict === "REAL").length}
              </p>
            </div>
            <div className="glass-panel p-5 rounded-2xl border border-amber-500/30 bg-amber-950/20 text-center">
              <span className="text-xs text-amber-400">Inconclusive</span>
              <p className="text-3xl font-extrabold text-amber-400 mt-1">
                {reports.filter((r) => r.verdict === "UNCERTAIN").length}
              </p>
            </div>
          </div>

          {/* Table Controls & Download */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {(["ALL", "FAKE", "REAL", "UNCERTAIN"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-semibold border transition ${
                    filter === f
                      ? "border-sky-400 bg-sky-500/20 text-white"
                      : "border-white/10 bg-white/5 text-slate-400 hover:text-white"
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>

            <button
              onClick={downloadCsv}
              className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/10 px-4 py-2 text-xs font-bold text-white hover:bg-white/20 transition"
            >
              <Download className="h-4 w-4" /> Download CSV Summary
            </button>
          </div>

          {/* Results Table */}
          <div className="glass-panel rounded-3xl overflow-hidden border border-white/10">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-white/10 bg-white/5 text-slate-300 uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="p-4">File Name</th>
                  <th className="p-4">Verdict</th>
                  <th className="p-4">Confidence</th>
                  <th className="p-4">Type</th>
                  <th className="p-4">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-slate-200 font-mono">
                {filteredReports.map((r, i) => {
                  const isF = r.verdict === "FAKE";
                  const isR = r.verdict === "REAL";

                  return (
                    <tr key={i} className="hover:bg-white/5 transition">
                      <td className="p-4 font-sans font-medium text-white truncate max-w-xs">
                        {r.metadata.original_filename || r.metadata.file_name || `Item ${i + 1}`}
                      </td>
                      <td className="p-4">
                        <span
                          className={`rounded-full px-2.5 py-1 text-[10px] font-bold uppercase border ${
                            isF
                              ? "border-red-500/40 bg-red-500/10 text-red-400"
                              : isR
                              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                              : "border-amber-500/40 bg-amber-500/10 text-amber-400"
                          }`}
                        >
                          {r.verdict}
                        </span>
                      </td>
                      <td className="p-4 font-bold text-white">{(r.confidence * 100).toFixed(1)}%</td>
                      <td className="p-4 uppercase text-slate-400 font-sans">{r.media_type}</td>
                      <td className="p-4 text-slate-400">{r.processing_time_ms.toFixed(0)} ms</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  );
}
