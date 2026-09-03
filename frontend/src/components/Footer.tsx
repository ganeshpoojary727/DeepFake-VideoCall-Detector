"use client";

import { Shield, FileText, GitBranch, ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="relative z-10 mt-24 border-t border-white/[0.06]">
      {/* Ambient glow line */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/3 h-px bg-gradient-to-r from-transparent via-[#38bdf8]/30 to-transparent" />

      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-tr from-[#38bdf8] to-[#6366f1]">
                <Shield className="h-4 w-4 text-white" />
              </div>
              <span className="text-sm font-bold tracking-tight text-white">
                DeepGuard Forensics
              </span>
            </div>
            <p className="text-xs text-[#6b7280] leading-relaxed max-w-[260px]">
              Enterprise-grade deepfake detection powered by AASIST, EfficientNet-B4, and multimodal forensic fusion.
            </p>
          </div>

          {/* Documentation */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-[#9ca3af]">
              Documentation
            </h4>
            <ul className="space-y-2">
              {[
                { label: "API Reference", href: "http://localhost:8000/docs" },
                { label: "ReDoc", href: "http://localhost:8000/redoc" },
                { label: "Whitepaper", href: "#" },
                { label: "Architecture Guide", href: "#" },
              ].map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 text-xs text-[#6b7280] hover:text-[#38bdf8] transition-colors"
                  >
                    <ExternalLink className="h-3 w-3 opacity-50" />
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Model Weights */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-[#9ca3af]">
              Model Checkpoints
            </h4>
            <div className="space-y-2">
              {[
                { name: "AASIST (ASVspoof5)", status: "Active" },
                { name: "EfficientNet-B4+TF", status: "Active" },
                { name: "YuNet ONNX", status: "Loaded" },
                { name: "Multimodal Fusion", status: "0.6A+0.4V" },
              ].map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between text-xs"
                >
                  <span className="text-[#6b7280]">{model.name}</span>
                  <span className="font-mono text-[10px] text-[#10b981] bg-[#10b981]/10 px-1.5 py-0.5 rounded">
                    {model.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Compliance */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-[#9ca3af]">
              Compliance & Privacy
            </h4>
            <p className="text-xs text-[#6b7280] leading-relaxed">
              All media processing is performed locally on-device. No data is transmitted to external servers. Analysis artifacts are ephemeral and not persisted.
            </p>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs text-[#6b7280] hover:text-white transition-colors mt-2"
            >
              <GitBranch className="h-3.5 w-3.5" />
              <span>View Source</span>
            </a>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-10 pt-6 border-t border-white/[0.04] flex flex-col md:flex-row items-center justify-between gap-3">
          <p className="text-[11px] text-[#6b7280]">
            © {new Date().getFullYear()} DeepGuard Forensics. Research & Development Platform.
          </p>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-[11px] font-mono text-[#6b7280]">
              <FileText className="h-3 w-3" />
              AASIST · EfficientNet-B4 · YuNet · 2D FFT · ELA · SRM
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
