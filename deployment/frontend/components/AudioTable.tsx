"use client";

import { PredictionResponse } from "@/types";

interface AudioTableProps {
  predictions: PredictionResponse[];
}

export default function AudioTable({ predictions }: AudioTableProps) {
  if (predictions.length === 0) {
    return (
      <div className=" text-center text-gray-400 py-8 rounded-2xl">
        No hay predicciones aún. Sube un audio para comenzar.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-2xl">
      <table className="w-full text-sm text-left text-gray-200">
        <thead className="box blue-box text-gray-300 uppercase text-xs">
          <tr>
            <th className="px-4 py-4">Archivo</th>
            <th className="px-4 py-4">¿Alertable?</th>
            <th className="px-4 py-4">Clase Predicha</th>
            <th className="px-4 py-4">Confianza</th>
            <th className="px-4 py-4">Conf. Binaria</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-white/10">
          {predictions.map((pred, idx) => (
            <tr
              key={idx}
              className="transition-all duration-200 hover:bg-white/5"
            >
              <td className="px-4 py-4 font-medium text-gray-100 max-w-xs truncate">
                {pred.filename}
              </td>

              <td className="px-4 py-4">
                <span
                  className={`px-3 py-1 rounded-full text-xs font-semibold border ${pred.is_alertable
                    ? "bg-red-500/15 text-red-300 border-red-400/20"
                    : "bg-green-500/15 text-green-300 border-green-400/20"
                    }`}
                >
                  {pred.is_alertable
                    ? "🚨 Alertable"
                    : "✅ No Alertable"}
                </span>
              </td>

              <td className="px-4 py-4 text-gray-300 capitalize">
                {pred.prediction.replace(/_/g, " ")}
              </td>

              <td className="px-4 py-4">
                <div className="flex items-center gap-3">
                  <div className="w-20 h-2 rounded-full overflow-hidden box blue-box-noafter">
                    <div
                      className="h-2 rounded-full bg-blue-500 transition-all duration-300"
                      style={{
                        width: `${(pred.confidence * 100).toFixed(0)}%`,
                      }}
                    />
                  </div>

                  <span className="text-gray-300 text-xs font-medium">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </td>

              <td className="px-4 py-4 text-gray-300 font-medium">
                {(pred.binary_confidence * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}