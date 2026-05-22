"use client";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip
} from "recharts";

const metricsData = [
  { metric: "Accuracy", value: 94.1 },
  { metric: "Precision", value: 92.8 },
  { metric: "Recall", value: 91.5 },
  { metric: "F1-Score", value: 92.1 },
  { metric: "AUC-ROC", value: 97.3 },
];

const modelInfo = [
  { label: "Arquitectura", value: "HybridCNN v3 (mel + waveform)" },
  { label: "Modo Binario", value: "best_alertable_v3.pt" },
  { label: "Modo Alertable", value: "best_human_label_v6.pt" },
  { label: "Modo No Alertable", value: "best_no_alertable_v4.pt" },
  { label: "Sample Rate", value: "16,000 Hz" },
  { label: "Clases Alertables", value: "10 clases" },
  { label: "Clases No Alertables", value: "14 clases" },
  { label: "Parámetros", value: "~1.8M por modelo" },
];

export default function ModelMetrics() {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-gray-800">Métricas del Modelo</h2>

      {/* Radar chart */}
      <div className="bg-white rounded-xl shadow p-6">
        <h3 className="text-base font-semibold text-gray-700 mb-4">Rendimiento General (Modelo Binario)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={metricsData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
            <PolarRadiusAxis angle={30} domain={[80, 100]} tick={{ fontSize: 10 }} />
            <Radar
              name="Métricas"
              dataKey="value"
              stroke="#3b82f6"
              fill="#3b82f6"
              fillOpacity={0.3}
            />
            <Tooltip formatter={(v: number) => [`${v}%`]} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Tabla de métricas numéricas */}
      <div className="bg-white rounded-xl shadow p-6">
        <h3 className="text-base font-semibold text-gray-700 mb-4">Detalles del Modelo</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {modelInfo.map((item) => (
            <div key={item.label} className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-sm text-gray-500">{item.label}</span>
              <span className="text-sm font-medium text-gray-800">{item.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Tabla de métricas */}
      <div className="bg-white rounded-xl shadow p-6">
        <h3 className="text-base font-semibold text-gray-700 mb-4">Resumen de Métricas</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-4 py-2 text-left text-gray-600">Métrica</th>
              <th className="px-4 py-2 text-right text-gray-600">Valor</th>
              <th className="px-4 py-2 text-left text-gray-600">Barra</th>
            </tr>
          </thead>
          <tbody>
            {metricsData.map((m) => (
              <tr key={m.metric} className="border-t border-gray-100">
                <td className="px-4 py-2 font-medium text-gray-700">{m.metric}</td>
                <td className="px-4 py-2 text-right text-blue-600 font-bold">{m.value}%</td>
                <td className="px-4 py-2">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${m.value}%` }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
