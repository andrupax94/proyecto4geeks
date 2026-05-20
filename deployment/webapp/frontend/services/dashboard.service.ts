import { apiGet } from './api';

export type DashboardSummary = {
  status: string;
  message: string;
};

export function getDashboardSummary() {
  return apiGet<DashboardSummary>('/eda/summary');
}
