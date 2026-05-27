import type { Metadata } from "next";
import "./globals.css";
import localFont from "next/font/local";

export const centuryGothic = localFont({
  src: [
    {
      path: "./fonts/CenturyGothic-Regular.ttf",
      weight: "400",
      style: "normal",
    },
    {
      path: "./fonts/CenturyGothic-Italic.ttf",
      weight: "400",
      style: "italic",
    },
    {
      path: "./fonts/CenturyGothic-Bold.ttf",
      weight: "700",
      style: "normal",
    },
    {
      path: "./fonts/CenturyGothic-BoldItalic.ttf",
      weight: "700",
      style: "italic",
    },
  ],
  variable: "--font-secondary",
});
export const Forte = localFont({
  src: [
    {
      path: "./fonts/Forte Regular.ttf",
      weight: "bold",
      style: "normal",
    }

  ],
  variable: "--font-primary",
});

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
      <body className={`${centuryGothic.variable} ${Forte.variable} antialiased dark`}>{children}</body>
    </html>
  );
}
