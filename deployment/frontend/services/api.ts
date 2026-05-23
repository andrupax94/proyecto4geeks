import { PredictionResponse, DashboardStats, EDAData } from "@/types";

// Selección automática del backend según entorno
const API_URL =
  process.env.NODE_ENV === "development"
    ? process.env.LOCALAPI || "http://127.0.0.1:8000"
    : process.env.HTTPAPI || "http://andreseduardo.ddns.net:8000";

export async function predictAudio(formData: FormData): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/predict`, {
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
