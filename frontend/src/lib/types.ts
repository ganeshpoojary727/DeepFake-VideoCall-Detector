export interface TemporalSyncItem {
  second: number;
  audio_spoof_prob: number | null;
  visual_spoof_prob: number | null;
  fused_spoof_prob: number;
  is_anomaly: boolean;
}

export interface TopAnomaly {
  timestamp_sec: number;
  modality: "visual" | "audio" | "cross_modal" | string;
  description: string;
  anomaly_score: number;
}

export interface KeyArtifact {
  frame_idx: number;
  timestamp_sec: number;
  bbox: [number, number, number, number];
  spoof_prob: number;
  saliency_peak: [number, number];
}

export interface VisualCues {
  ela_discrepancy_score?: number;
  fft_spectral_anomaly?: number;
  boundary_inconsistency?: number;
  combined_score?: number;
  [key: string]: unknown;
}

export interface SpectralCues {
  spectral_rolloff_hz?: number;
  spectral_flatness?: number;
  high_freq_energy_ratio?: number;
  artifacts_detected?: string[];
  peak_artifact_ranges?: Array<{
    start_hz?: number;
    end_hz?: number;
    severity?: string;
    description?: string;
  }>;
  [key: string]: unknown;
}

export interface NaturalLanguageReport {
  executive_summary: string;
  visual_analysis_narrative: string;
  audio_analysis_narrative: string;
  temporal_inconsistency_notes: string;
  forensic_recommendations: string[];
  provider_used: string;
  generation_timestamp: string;
}

export interface ModalityBreakdown {
  audio?: {
    verdict?: "REAL" | "FAKE";
    confidence?: number;
    raw_scores?: {
      bonafide_prob?: number;
      spoof_prob?: number;
    };
    spectral_cues?: SpectralCues;
    timeline?: Array<{
      chunk_index?: number;
      start_time_sec?: number;
      end_time_sec?: number;
      spoof_prob?: number;
      verdict?: string;
    }>;
  } | null;
  visual?: {
    verdict?: "REAL" | "FAKE";
    confidence?: number;
    raw_scores?: {
      real_prob?: number;
      fake_prob?: number;
    };
    visual_cues?: VisualCues;
    timeline?: Array<{
      frame_idx?: number;
      timestamp_sec?: number;
      spoof_prob?: number;
      is_anomaly?: boolean;
    }>;
    key_artifacts?: KeyArtifact[];
  } | null;
  classical_forensics?: VisualCues & {
    spectral_cues?: SpectralCues;
  } | null;
}

export interface ConsolidatedForensicReport {
  media_type: "AUDIO" | "IMAGE" | "VIDEO" | "MULTIMODAL" | string;
  verdict: "REAL" | "FAKE" | "UNCERTAIN" | "NOT_APPLICABLE";
  overall_confidence: number; // 0.0 to 1.0
  content_category?: string | null;  // Set for NOT_APPLICABLE (e.g. "DIGITAL_ART_ANIME")
  modality_breakdown: ModalityBreakdown;
  temporal_sync: TemporalSyncItem[];
  top_anomalies: TopAnomaly[];
  natural_language_report?: NaturalLanguageReport | null;
  processing_time_ms: number;
  metadata: {
    file_name?: string;
    original_filename?: string;
    fused_spoof_probability?: number;
    audio_spoof_prob?: number;
    visual_spoof_prob?: number;
    anomaly_boost_applied?: boolean;
    error?: string;
    [key: string]: unknown;
  };
}

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
  verdict: "REAL" | "FAKE" | "UNCERTAIN" | "NOT_APPLICABLE";
  confidence: number;
  real_confidence: number;
  fake_confidence: number;
  media_type: "image" | "video" | "audio" | "unknown";
  content_category?: string | null;  // Set for NOT_APPLICABLE verdicts
  scores: {
    video?: number | null;
    audio?: number | null;
    fused?: number | null;
    image?: number | null;
    [key: string]: number | null | undefined;
  };
  processing_time_ms: number;
  metadata: Record<string, unknown>;
  forensics?: ForensicReport;
  consolidated?: ConsolidatedForensicReport;
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

/** Session-stored scan data for navigating between landing and report pages */
export interface ScanSession {
  scanId: string;
  fileName: string;
  fileSize: number;
  mediaType: string;
  timestamp: string;
  report: ConsolidatedForensicReport;
}

/** Display-ready diagnostic factor for the Evidence Matrix */
export interface DiagnosticFactorDisplay {
  name: string;
  score: number;
  status: "NATURAL" | "ANOMALOUS" | "UNCERTAIN";
  icon: string;
  description: string;
  color: "emerald" | "crimson" | "amber" | "cyan";
}
