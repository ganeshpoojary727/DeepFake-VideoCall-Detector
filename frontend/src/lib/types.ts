export interface DiagnosticFactor {
  name: string;
  score: number; // 0 - 100
  status: "ANOMALOUS" | "UNCERTAIN" | "NATURAL";
  description: string;
  details: string;
}

export interface ForensicReport {
  threat_level: "AUTHENTIC" | "CLEAN" | "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  diagnostic_factors: DiagnosticFactor[];
  narrative_conclusion: string;
  key_indicators: string[];
}

export interface AnalysisReport {
  verdict: "REAL" | "FAKE" | "UNCERTAIN";
  confidence: number; // 0.0 - 1.0 (verdict confidence)
  real_confidence: number; // 0.0 - 1.0
  fake_confidence: number; // 0.0 - 1.0
  media_type: "image" | "video" | "audio" | "unknown";
  scores: {
    video?: number | null;
    audio?: number | null;
    fused?: number | null;
    image?: number | null;
    [key: string]: number | null | undefined;
  };
  processing_time_ms: number;
  metadata: {
    file_name?: string;
    original_filename?: string;
    num_frames?: number;
    num_faces_detected?: number;
    face_bbox?: { x: number; y: number; w: number; h: number } | null;
    original_dimensions?: [number, number];
    sample_rate?: number;
    duration_seconds?: number;
    model?: string;
    forensic_signals?: Record<string, any>;
    error?: string;
    [key: string]: any;
  };
  forensics?: ForensicReport;
}

export interface SystemStatus {
  status: string;
  device: string;
  cuda_available: boolean;
  torch_version: string;
  gpu_name?: string;
  vram_allocated_mb?: number;
  vram_reserved_mb?: number;
  models: {
    audio_aasist?: { checkpoint_exists: boolean; path: string };
    video_efficientnet_transformer?: { checkpoint_exists: boolean; path: string };
  };
}
