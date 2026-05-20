import { apiGet } from './api';

export function getSounds() {
  return apiGet('/sounds');
}
