// Tipos para la respuesta de predicción del backend
export interface TopKItem {
  label: string;
  confidence: number;
}

export interface PredictionResponse {
  filename: string;
  is_alertable: boolean;
  binary_label: "alertable" | "no_alertable";
  binary_confidence: number;
  binary_top_k: TopKItem[];
  prediction: string;
  confidence: number;
  top_k: TopKItem[];
}

// Tipos para el dashboard
export interface DashboardStats {
  total_files: number;
  classes: number;
  alertable_count: number;
  no_alertable_count: number;
  model_accuracy: number;
}

export interface ClassDistributionItem {
  class: string;
  count: number;
}

export interface EDAData {
  alertable: ClassDistributionItem[];
  no_alertable: ClassDistributionItem[];
}
