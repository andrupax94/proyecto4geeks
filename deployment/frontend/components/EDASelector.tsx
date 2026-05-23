"use client";
import { motion } from "framer-motion";
import colors, {
  DARK_COLORS, getColor
} from '@/services/colors';
interface EDASelectorProps {
  activeChart: string;
  onChartChange: (chart: string) => void;
}

const charts = [
  { id: "alertable", label: "Clases Alertables", icon: "🚨" },
  { id: "no_alertable", label: "Clases No Alertables", icon: "✅" },
  { id: "source", label: "Fuente de Datos", icon: "📦" },
  { id: "format", label: "Formato de Audio", icon: "🎵" },
  { id: "duration", label: "Duración de Audios", icon: "⏱️" },
  { id: "sample_rate", label: "Frecuencia de Muestreo", icon: "📊" },
];

export default function EDASelector({ activeChart, onChartChange }: EDASelectorProps) {
  return (
    <div style={{ backgroundColor: getColor("primary") }} className="bg-white rounded-xl shadow p-6 mb-6">
      <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
        Selecciona un análisis
      </h3>
      <div className="flex flex-wrap gap-3">
        {charts.map((chart) => (
          <motion.button
            key={chart.id}
            onClick={() => onChartChange(chart.id)}
            className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${activeChart === chart.id
              ? "bg-blue-600 text-white shadow-lg"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            <span className="text-lg">{chart.icon}</span>
            <span className="text-sm">{chart.label}</span>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
