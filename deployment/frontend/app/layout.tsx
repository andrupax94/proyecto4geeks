import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MIVIA - Sistema de Detección de sonidos",
  description: "Monitoreo Inteligente de Vigilancia con Inteligencia Artificial",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body className="antialiased">{children}</body>
    </html>
  );
}
