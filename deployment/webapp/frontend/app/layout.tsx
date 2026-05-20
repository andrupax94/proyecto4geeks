import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'DataScience Multifunction',
  description: 'Scaffold for EDA, model, wiki and mobile view dashboard',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
