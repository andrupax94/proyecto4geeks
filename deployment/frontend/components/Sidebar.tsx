"use client";
import { useState } from "react";

interface SidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

const sections = [
  { id: "dashboard", label: "Dashboard", icon: "📊" },
  { id: "predict", label: "Predicción", icon: "🎙️" },
  { id: "eda", label: "Análisis EDA", icon: "📈" },
  { id: "metrics", label: "Métricas del Modelo", icon: "🧠" },
  { id: "wiki", label: "Wiki del Proyecto", icon: "📚" },
  { id: "edas", label: "EDAs", icon: "📊" },
];

export default function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  return (
    <aside className="w-64 min-h-screen bg-gray-900 text-white flex flex-col py-6 px-4 gap-2 shadow-lg">
      <div className="mb-6 px-2">
        <h1 className="text-xl font-bold text-blue-400">MIVIA</h1>
        <p className="text-xs text-gray-500 mt-1">Detección de Sonidos</p>
      </div>
      {sections.map((section) => (
        <button
          key={section.id}
          onClick={() => onSectionChange(section.id)}
          className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
            activeSection === section.id
              ? "bg-blue-600 text-white"
              : "text-gray-300 hover:bg-gray-800 hover:text-white"
          }`}
        >
          <span>{section.icon}</span>
          <span>{section.label}</span>
        </button>
      ))}
    </aside>
  );
}
