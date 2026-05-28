// Tipos para la respuesta de predicción del backend
export interface TopKItem {
  label: string;
  confidence: number;
}
export interface TrainingHistory {
  epoch: number[];

  train_loss: number[];
  val_loss: number[];

  train_acc: number[];
  val_acc: number[];

  precision: number[];
  recall: number[];
  f1: number[];

  auc_roc?: number[];

  lr: number[];

  epoch_time: number[];
  images_per_sec: number[];
}

export interface TrainingData {
  history: TrainingHistory;

  best_epoch: number;
  best_acc: number;

  model_name?: string;
  version?: number;
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
  f1_macro: number | null;
  f1_weighted: number | null;
  multiclass?: {
    accuracy: number;
    f1_macro: number;
    f1_weighted: number;
  };
  class_report?: Record<string, {
    precision: number;
    recall: number;
    "f1-score": number;
    support: number;
  }>;
}

export interface ClassDistributionItem {
  class: string;
  count: number;
}

export interface SourceDistributionItem {
  source: string;
  count: number;
}

export interface FormatDistributionItem {
  format: string;
  count: number;
}

export interface SampleRateDistributionItem {
  rate: number;
  count: number;
}

export interface DurationStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  std: number;
}

export interface EDAData {
  alertable: ClassDistributionItem[];
  no_alertable: ClassDistributionItem[];
  dataset_source_distribution: SourceDistributionItem[];
  audio_format_distribution: FormatDistributionItem[];
  duration_stats: DurationStats;
  sample_rate_distribution: SampleRateDistributionItem[];
}