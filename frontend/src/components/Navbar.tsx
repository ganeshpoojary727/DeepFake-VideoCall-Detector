"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Shield, Cpu, ExternalLink, Scan } from "lucide-react";
import { getSystemHealth } from "../lib/api";
import { SystemStatus } from "../lib/types";

export default function Navbar() {
  const [health, setHealth] = useState<SystemStatus | null>(null);

  useEffect(() => {
    getSystemHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/[0.06] bg-[#07080f]/85 backdrop-blur-2xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        {/* ── Brand ── */}
        <motion.a
          href="/"
          initial={{ opacity: 0, x: -15 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-3 cursor-pointer no-underline"
        >
          <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-[#38bdf8] to-[#6366f1] shadow-[0_0_24px_rgba(56,189,248,0.25)]">
            <Shield className="h-4.5 w-4.5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-extrabold tracking-tight text-white">
                DeepGuard
              </span>
              <span className="text-sm font-light text-[#9ca3af]">
                Forensics
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#10b981] opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-[#10b981]" />
              </span>
              <span className="text-[10px] font-mono text-[#10b981]">
                {health?.cuda_available ? "CUDA Engine Active" : "CPU Inference"}
              </span>
            </div>
          </div>
        </motion.a>

        {/* ── Nav Links ── */}
        <nav className="hidden md:flex items-center gap-1">
          {[
            { label: "Capabilities", target: "capabilities" },
            { label: "Detection Pipeline", target: "pipeline" },
            { label: "Benchmarks", target: "benchmarks" },
          ].map((link) => (
            <button
              key={link.target}
              onClick={() => scrollToSection(link.target)}
              className="px-3.5 py-1.5 text-xs font-medium text-[#9ca3af] hover:text-white transition-colors rounded-lg hover:bg-white/[0.04]"
            >
              {link.label}
            </button>
          ))}
          <a
            href="/live"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-red-500/15 border border-red-500/30 rounded-lg hover:bg-red-500/25 transition-all shadow-[0_0_15px_rgba(239,68,68,0.2)]"
          >
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
            </span>
            Live Call Guard
          </a>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-[#9ca3af] hover:text-white transition-colors rounded-lg hover:bg-white/[0.04]"
          >
            API Docs
            <ExternalLink className="h-3 w-3 opacity-60" />
          </a>
        </nav>

        {/* ── Right Section: GPU Status & CTA ── */}
        <div className="flex items-center gap-3">
          {/* Hardware Status Pill */}
          <div className="hidden lg:flex items-center gap-2 rounded-full border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-xs text-[#9ca3af]">
            <Cpu className="h-3.5 w-3.5 text-[#38bdf8]" />
            <span className="font-mono text-[11px]">
              {health?.gpu_name
                ? health.gpu_name.replace("NVIDIA ", "")
                : "AASIST + EffNet-B4"}
            </span>
          </div>

          {/* CTA Button */}
          <button
            onClick={() => scrollToSection("scanner")}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#38bdf8] to-[#6366f1] px-4 py-2 text-xs font-bold text-white shadow-[0_0_20px_rgba(56,189,248,0.25)] hover:shadow-[0_0_30px_rgba(56,189,248,0.40)] transition-shadow"
          >
            <Scan className="h-3.5 w-3.5" />
            Launch Forensic Scan
          </button>
        </div>
      </div>
    </header>
  );
}
