import { apiGet } from './api';

export type WikiPage = {
  slug: string;
  title: string;
  body: string;
};

export function getWikiPages() {
  return apiGet<WikiPage[]>('/wiki/pages');
}
