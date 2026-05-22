import { PredictionResponse, DashboardStats, EDAData } from "@/types";

// En producción, configurar NEXT_PUBLIC_API_URL en .env.local
// En desarrollo local: http://127.0.0.1:8000
// En el sandbox público: usar la URL expuesta del backend
const API_URL = "http://127.0.0.1:8000";

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
