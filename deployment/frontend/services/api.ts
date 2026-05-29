import { PredictionResponse, DashboardStats, EDAData, TrainingData } from "@/types";


export interface Commit {
  hash: string;
  hash_full: string;
  message: string;
  author: string;
  email: string;
  date: string;
  url: string;
  files_changed: number;
  additions: number;
  deletions: number;
}

// Selección automática del backend según entorno
const isHttps =
  typeof window !== "undefined" &&
  window.location.protocol === "https:";
const API_URL =
  process.env.NODE_ENV === "development"
    ? process.env.NEXT_PUBLIC_LOCALAPI || "http://127.0.0.1:8000"
    : isHttps
      ? "https://andreseduardo.ddns.net/api"
      : "http://andreseduardo.ddns.net/api";
const versionBinary = 4
const versionMultiClass = 7
export async function getGithubCommits(limit = 12): Promise<Commit[]> {
  const response = await fetch(
    `${API_URL}/github/commits?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error("Error al obtener commits");
  }

  const data = await response.json();

  return data.commits;
}
export async function predictAudio(
  formData: FormData,
  versionBin: number = versionBinary,
  versionSpecific: number = versionMultiClass
): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/predict?version_bin=${versionBin}&version_specific=${versionSpecific}`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Error en predicción: ${response.status} - ${text}`);
  }
  return response.json();
}

export async function getStats(): Promise<DashboardStats> {
  const response = await fetch(`${API_URL}/dashboard/stats`);
  if (!response.ok) {
    throw new Error(`Error obteniendo stats: ${response.statusText}`);
  }
  return response.json();
}

export async function getEDA(): Promise<EDAData> {
  const response = await fetch(`${API_URL}/dashboard/eda`);
  if (!response.ok) {
    throw new Error(`Error obteniendo EDA: ${response.statusText}`);
  }
  return response.json();
}

export async function getTrainingHistory(targetType: string = "alertable", version: number = 4): Promise<TrainingData> {
  if (targetType == "alertable") {
    version = versionBinary;
  }
  else {
    version = versionMultiClass;
  }
  const response = await fetch(`${API_URL}/training/history?target_type=${targetType}&version=${version}`);
  if (!response.ok) {
    throw new Error(`Error obteniendo historial de entrenamiento: ${response.statusText}`);
  }
  return response.json();
}
