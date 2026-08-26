"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Shield, Cpu, Activity, ExternalLink } from "lucide-react";
import { getSystemHealth } from "../lib/api";
import { SystemStatus } from "../lib/types";

interface NavbarProps {
  activeTab: "single" | "batch" | "system";
  setActiveTab: (tab: "single" | "batch" | "system") => void;
}

export default function Navbar({ activeTab, setActiveTab }: NavbarProps) {
  const [health, setHealth] = useState<SystemStatus | null>(null);

  useEffect(() => {
    getSystemHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/10 bg-[#06070d]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        {/* Brand Logo & Name */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-3 cursor-pointer"
          onClick={() => setActiveTab("single")}
        >
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-sky-600 via-indigo-600 to-purple-600 shadow-[0_0_20px_rgba(56,189,248,0.4)]">
            <Shield className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-extrabold tracking-tight text-white">DEEPGUARD</span>
              <span className="rounded-full bg-sky-500/10 px-2 py-0.5 text-[10px] font-semibold text-sky-400 border border-sky-500/20">
                v2.0 XAI
              </span>
            </div>
            <p className="text-[11px] font-medium text-slate-400">Forensic Deepfake Intelligence</p>
          </div>
        </motion.div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 p-1 backdrop-blur-md">
          {[
            { id: "single", label: "Single File Inspector" },
            { id: "batch", label: "Batch Scanner" },
            { id: "system", label: "Neural Telemetry" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`relative rounded-full px-4 py-1.5 text-xs font-semibold transition-all duration-300 ${
                activeTab === tab.id ? "text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {activeTab === tab.id && (
                <motion.div
                  layoutId="activeTabBadge"
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-sky-500 to-indigo-600 shadow-[0_0_15px_rgba(56,189,248,0.4)]"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.5 }}
                />
              )}
              <span className="relative z-10">{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Hardware Status Pill & API Docs */}
        <div className="flex items-center gap-4">
          <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-300 md:flex">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
            </span>
            <Cpu className="h-3.5 w-3.5 text-sky-400" />
            <span className="font-mono text-[11px]">
              {health?.gpu_name ? health.gpu_name.replace("NVIDIA ", "") : "CUDA Engine Ready"}
            </span>
          </div>

          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-sky-500/40 hover:bg-white/10 hover:text-white"
          >
            <span>Swagger API</span>
            <ExternalLink className="h-3 w-3 opacity-70" />
          </a>
        </div>
      </div>
    </header>
  );
}
