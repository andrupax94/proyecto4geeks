import { apiGet } from './api';

export function getModelSummary() {
  return apiGet('/model/summary');
}
