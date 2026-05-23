"use client";
import { useState } from "react";

interface SidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

// 1. Estructuramos las secciones. 'wiki' ahora tiene 'subSections'.
const sections = [
  { id: "dashboard", label: "Dashboard", icon: "📊" },
  { id: "predict", label: "Predicción", icon: "🎙️" },
  { id: "eda", label: "Análisis EDA", icon: "📈" },
  { id: "metrics", label: "Métricas del Modelo", icon: "🧠" },
  {
    id: "wiki",
    label: "Wiki del Proyecto",
    icon: "📚",
    subSections: [
      { id: "wiki", label: "Acerca de", icon: "ℹ️" },
      { id: "edas", label: "EDAs", icon: "🔍" },
      { id: "preprocesado-y-modelado", label: "Preprocesado y modelado", icon: "⚙️" },
    ],
  },
];

export default function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  // 2. Estado para controlar si el menú de la Wiki está abierto
  // Lo inicializamos en true si la sección activa ya pertenece a la wiki
  const [isWikiOpen, setIsWikiOpen] = useState(() => {
    return ["wiki", "edas", "preprocesado-y-modelado"].includes(activeSection);
  });

  return (
    <aside className="w-64 min-h-screen bg-gray-900 text-white flex flex-col py-6 px-4 gap-2 shadow-lg">
      <div className="mb-6 px-2">
        <h1 className="text-xl font-bold text-blue-400">MIVIA</h1>
        <p className="text-xs text-gray-500 mt-1">Detección de Sonidos</p>
      </div>

      {sections.map((section) => {
        // Clonamos la lógica para saber si el botón principal o uno de sus hijos está activo
        const isSubActive = section.subSections?.some(sub => sub.id === activeSection);
        const isActive = activeSection === section.id || isSubActive;

        // Si la sección tiene subsecciones (caso de la Wiki)
        if (section.subSections) {
          return (
            <div key={section.id} className="flex flex-col gap-1">
              {/* Botón Principal del Dropdown */}
              <button
                onClick={() => setIsWikiOpen(!isWikiOpen)}
                className={`flex items-center justify-between w-full px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                  isActive && !isWikiOpen
                    ? "bg-blue-600 text-white"
                    : "text-gray-300 hover:bg-gray-800 hover:text-white"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span>{section.icon}</span>
                  <span>{section.label}</span>
                </div>
                {/* Flechita indicadora que rota si está abierto */}
                <span className={`text-xs transition-transform duration-200 ${isWikiOpen ? "rotate-180" : ""}`}>
                  ▼
                </span>
              </button>

              {/* Contenedor de Subsecciones */}
              {isWikiOpen && (
                <div className="flex flex-col gap-1 pl-6 mt-1 border-l border-gray-800 ml-4">
                  {section.subSections.map((sub) => (
                    <button
                      key={sub.id}
                      onClick={() => onSectionChange(sub.id)}
                      className={`flex items-center gap-3 text-left px-4 py-2 rounded-lg text-xs font-medium transition-all ${
                        activeSection === sub.id
                          ? "bg-blue-600/30 text-blue-400 font-semibold"
                          : "text-gray-400 hover:bg-gray-800 hover:text-white"
                      }`}
                    >
                      <span>{sub.icon}</span>
                      <span>{sub.label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        }

        {/* Botones normales sin subsecciones */}
        return (
          <button
            key={section.id}
            onClick={() => onSectionChange(section.id)}
            className={`flex items-center gap-3 text-left px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeSection === section.id
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-800 hover:text-white"
            }`}
          >
            <span>{section.icon}</span>
            <span>{section.label}</span>
          </button>
        );
      })}
    </aside>
  );
}