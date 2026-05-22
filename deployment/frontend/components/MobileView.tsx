"use client";
import { PredictionResponse } from "@/types";

interface MobileViewProps {
  lastPrediction: PredictionResponse | null;
  isLoading: boolean;
}

export default function MobileView({ lastPrediction, isLoading }: MobileViewProps) {
  return (
    <div className="w-72 hidden lg:flex flex-col items-center py-8 px-4 border-l border-gray-200 bg-gray-50">
      <p className="text-xs text-gray-400 uppercase tracking-widest mb-4">Vista Móvil</p>
      {/* Phone frame */}
      <div className="border-4 border-gray-800 rounded-3xl w-56 h-[500px] bg-white shadow-2xl overflow-hidden flex flex-col">
        {/* Status bar */}
        <div className="bg-gray-900 text-white text-xs px-4 py-2 flex justify-between items-center">
          <span>MIVIA</span>
          <span>🔋 100%</span>
        </div>
        {/* Content */}
        <div className="flex-1 flex flex-col items-center justify-center p-4 gap-4">
          {isLoading ? (
            <div className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-gray-500">Analizando audio...</p>
            </div>
          ) : lastPrediction ? (
            <div className="w-full space-y-3">
              <div
                className={`rounded-xl p-3 text-center ${
                  lastPrediction.is_alertable
                    ? "bg-red-50 border border-red-200"
                    : "bg-green-50 border border-green-200"
                }`}
              >
                <p className="text-2xl mb-1">
                  {lastPrediction.is_alertable ? "🚨" : "✅"}
                </p>
                <p
                  className={`text-sm font-bold ${
                    lastPrediction.is_alertable ? "text-red-700" : "text-green-700"
                  }`}
                >
                  {lastPrediction.is_alertable ? "ALERTABLE" : "NO ALERTABLE"}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {(lastPrediction.binary_confidence * 100).toFixed(1)}% confianza
                </p>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-xl p-3 text-center">
                <p className="text-xs text-gray-500 uppercase tracking-wide">Clase detectada</p>
                <p className="text-base font-bold text-blue-800 capitalize mt-1">
                  {lastPrediction.prediction.replace(/_/g, " ")}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {(lastPrediction.confidence * 100).toFixed(1)}% confianza
                </p>
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-3">
                <p className="text-xs text-gray-500 mb-2">Top 3 clases:</p>
                {lastPrediction.top_k.slice(0, 3).map((item, i) => (
                  <div key={i} className="flex justify-between text-xs py-0.5">
                    <span className="text-gray-700 capitalize">{item.label.replace(/_/g, " ")}</span>
                    <span className="text-blue-600 font-medium">
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 text-center">
              <span className="text-5xl">🎙️</span>
              <p className="text-sm text-gray-500">
                Sube un archivo de audio para ver la predicción aquí
              </p>
            </div>
          )}
        </div>
        {/* Bottom bar */}
        <div className="bg-gray-100 px-4 py-2 text-center">
          <p className="text-xs text-gray-400">MIVIA v1.0</p>
        </div>
      </div>
    </div>
  );
}
