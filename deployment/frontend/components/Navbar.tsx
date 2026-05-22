"use client";
import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="bg-gray-900 text-white px-6 py-3 flex items-center justify-between shadow-md">
      <div className="flex items-center gap-3">
        <span className="text-2xl font-bold text-blue-400">MIVIA</span>
        <span className="text-sm text-gray-400">Sistema de Detección de Sonidos</span>
      </div>
      <div className="flex gap-6 text-sm">
        <Link href="/" className="hover:text-blue-400 transition-colors">Dashboard</Link>
        <Link href="#predict" className="hover:text-blue-400 transition-colors">Predicción</Link>
        <Link href="#eda" className="hover:text-blue-400 transition-colors">EDA</Link>
        <Link href="#wiki" className="hover:text-blue-400 transition-colors">Wiki</Link>
      </div>
    </nav>
  );
}
