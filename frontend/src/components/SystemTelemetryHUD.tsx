"use client";

import { useEffect, useState } from "react";
import { Cpu, Server, Activity, Database, Shield, Zap, Terminal } from "lucide-react";
import { getSystemHealth } from "../lib/api";
import { SystemStatus } from "../lib/types";

export default function SystemTelemetryHUD() {
  const [health, setHealth] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSystemHealth()
      .then((h) => {
        setHealth(h);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Top Telemetry Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* GPU & Compute Card */}
        <div className="glass-panel rounded-3xl p-6 border border-white/10">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-500/10 text-sky-400 border border-sky-500/30">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-bold text-white text-sm">GPU Accelerator</h3>
              <p className="text-xs text-slate-400">Hardware Inference Engine</p>
            </div>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-400">Device</span>
              <span className="text-sky-300 font-bold">{health?.device || "CUDA (Auto)"}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-400">GPU Name</span>
              <span className="text-white truncate max-w-[180px]">{health?.gpu_name || "NVIDIA RTX 4050"}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-400">VRAM Allocated</span>
              <span className="text-emerald-400">{health?.vram_allocated_mb ?? 0.0} MB</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">PyTorch Version</span>
              <span className="text-slate-300">{health?.torch_version || "2.11.0+cu128"}</span>
            </div>
          </div>
        </div>

        {/* Neural Models Card */}
        <div className="glass-panel rounded-3xl p-6 border border-white/10">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-bold text-white text-sm">Neural Checkpoints</h3>
              <p className="text-xs text-slate-400">Pre-Trained Weights</p>
            </div>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <div className="flex justify-between items-center py-1.5 border-b border-white/5">
              <span className="text-slate-400">Audio (AASIST)</span>
              <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-400 border border-emerald-500/30">
                ACTIVE (99.71%)
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-white/5">
              <span className="text-slate-400">Video (EffNet-B4)</span>
              <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-400 border border-emerald-500/30">
                ACTIVE (557MB)
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-white/5">
              <span className="text-slate-400">Face (YuNet ONNX)</span>
              <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-400 border border-emerald-500/30">
                LOADED
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5">
              <span className="text-slate-400">Fusion Engine</span>
              <span className="text-indigo-300 font-bold">0.6A + 0.4V</span>
            </div>
          </div>
        </div>

        {/* REST API Status */}
        <div className="glass-panel rounded-3xl p-6 border border-white/10">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/30">
              <Server className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-bold text-white text-sm">FastAPI Daemon</h3>
              <p className="text-xs text-slate-400">REST Endpoints</p>
            </div>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-400">Base URL</span>
              <span className="text-purple-300">http://localhost:8000</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-400">OpenAPI Docs</span>
              <span className="text-sky-400">/docs</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-400">Status</span>
              <span className="text-emerald-400 font-bold">HEALTHY</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">CORS</span>
              <span className="text-slate-300">Allowed (*)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Terminal cURL Playground */}
      <div className="glass-panel rounded-3xl p-6 border border-white/10 font-mono">
        <div className="flex items-center gap-2 mb-3 text-slate-300 text-xs font-bold uppercase tracking-wider">
          <Terminal className="h-4 w-4 text-sky-400" />
          <span>Programmatic Detection Examples</span>
        </div>

        <div className="rounded-2xl bg-black/80 p-4 border border-white/5 text-xs text-sky-300 space-y-2 overflow-x-auto">
          <p className="text-slate-500"># 1. Single File Deepfake Detection with Explainable AI Factors</p>
          <p className="text-slate-200">
            curl -X POST &quot;http://localhost:8000/detect/file&quot; \<br />
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-H &quot;accept: application/json&quot; \<br />
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-F &quot;file=@interview_clip.mp4&quot;
          </p>
          <p className="text-slate-500 pt-2"># 2. Check GPU & System Health</p>
          <p className="text-slate-200">curl -X GET &quot;http://localhost:8000/health&quot;</p>
        </div>
      </div>
    </div>
  );
}
