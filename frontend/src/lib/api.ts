import { AnalysisReport, SystemStatus } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Upload and analyze a single media file (Image, Video, Audio)
 */
export async function analyzeMediaFile(file: File): Promise<AnalysisReport> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/detect/file`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server responded with status ${response.status}`);
  }

  return response.json();
}

/**
 * Upload and analyze multiple media files in batch
 */
export async function analyzeMediaBatch(files: File[]): Promise<AnalysisReport[]> {
  const formData = new FormData();
  for (const f of files) {
    formData.append("files", f);
  }

  const response = await fetch(`${API_BASE_URL}/detect/batch`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Batch analysis failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch hardware status, CUDA GPU metrics, and model readiness
 */
export async function getSystemHealth(): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error("Unable to fetch system health");
  }
  return response.json();
}
