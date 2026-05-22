"use client";
import { PredictionResponse } from "@/types";

interface AudioTableProps {
  predictions: PredictionResponse[];
}

export default function AudioTable({ predictions }: AudioTableProps) {
  if (predictions.length === 0) {
    return (
      <div className="text-center text-gray-400 py-8">
        No hay predicciones aún. Sube un audio para comenzar.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead className="bg-gray-100 text-gray-600 uppercase text-xs">
          <tr>
            <th className="px-4 py-3">Archivo</th>
            <th className="px-4 py-3">¿Alertable?</th>
            <th className="px-4 py-3">Clase Predicha</th>
            <th className="px-4 py-3">Confianza</th>
            <th className="px-4 py-3">Conf. Binaria</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {predictions.map((pred, idx) => (
            <tr key={idx} className="hover:bg-gray-50 transition-colors">
              <td className="px-4 py-3 font-medium text-gray-800 max-w-xs truncate">
                {pred.filename}
              </td>
              <td className="px-4 py-3">
                <span
                  className={`px-2 py-1 rounded-full text-xs font-semibold ${
                    pred.is_alertable
                      ? "bg-red-100 text-red-700"
                      : "bg-green-100 text-green-700"
                  }`}
                >
                  {pred.is_alertable ? "🚨 Alertable" : "✅ No Alertable"}
                </span>
              </td>
              <td className="px-4 py-3 text-gray-700 capitalize">
                {pred.prediction.replace(/_/g, " ")}
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${(pred.confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="text-gray-600">{(pred.confidence * 100).toFixed(1)}%</span>
                </div>
              </td>
              <td className="px-4 py-3 text-gray-600">
                {(pred.binary_confidence * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
