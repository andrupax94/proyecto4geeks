"use client";
import { useState, useEffect, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import DashboardCard from "@/components/DashboardCard";
import EDAChart from "@/components/EDAChart";
import AudioTable from "@/components/AudioTable";
import ModelMetrics from "@/components/ModelMetrics";
import WikiSection from "@/components/WikiSection";
import EDAS from "@/components/EDAS";
import Preprocessing from "@/components/Preprocessing";
import MobileView from "@/components/MobileView";
import { predictAudio, getStats, getEDA } from "@/services/api";
import { PredictionResponse, DashboardStats, EDAData } from "@/types";

export default function Home() {
  const [activeSection, setActiveSection] = useState("dashboard");
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [edaData, setEdaData] = useState<EDAData | null>(null);
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [lastPrediction, setLastPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getStats().then(setStats).catch(console.error);
    getEDA().then(setEdaData).catch(console.error);
  }, []);

  const handleFile = async (file: File) => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const result = await predictAudio(formData);
      setPredictions((prev) => [result, ...prev]);
      setLastPrediction(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error desconocido al predecir");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const renderSection = () => {
    switch (activeSection) {
      case "dashboard":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-800">Dashboard</h2>
            {stats ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <DashboardCard
                  title="Total Audios"
                  value={stats.total_files.toLocaleString()}
                  subtitle="Dataset de entrenamiento"
                  color="blue"
                  icon="🎵"
                />
                <DashboardCard
                  title="Clases"
                  value={stats.classes}
                  subtitle="Categorías de sonido"
                  color="purple"
                  icon="🏷️"
                />
                <DashboardCard
                  title="Alertables"
                  value={stats.alertable_count.toLocaleString()}
                  subtitle="Sonidos de alerta"
                  color="red"
                  icon="🚨"
                />
                <DashboardCard
                  title="Accuracy"
                  value={`${(stats.model_accuracy * 100).toFixed(1)}%`}
                  subtitle="Modelo binario"
                  color="green"
                  icon="🎯"
                />
              </div>
            ) : (
              <div className="text-gray-400 text-sm">Cargando estadísticas...</div>
            )}
            {/* Historial de predicciones */}
            <div className="bg-white rounded-xl shadow p-5">
              <h3 className="text-base font-semibold text-gray-700 mb-4">
                Historial de predicciones
              </h3>
              <AudioTable predictions={predictions} />
            </div>
          </div>
        );

      case "predict":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-800">Predicción de Audio</h2>
            {/* Upload area */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${
                dragOver
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.ogg,.flac,.m4a"
                className="hidden"
                onChange={handleFileChange}
              />
              <div className="flex flex-col items-center gap-3">
                <span className="text-5xl">{isLoading ? "⏳" : "🎙️"}</span>
                {isLoading ? (
                  <p className="text-blue-600 font-medium">Analizando audio con el modelo...</p>
                ) : (
                  <>
                    <p className="text-gray-600 font-medium">
                      Arrastra un archivo de audio aquí o haz clic para seleccionar
                    </p>
                    <p className="text-xs text-gray-400">
                      Formatos soportados: .wav, .mp3, .ogg, .flac, .m4a
                    </p>
                  </>
                )}
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
                ❌ {error}
              </div>
            )}

            {/* Resultado de la última predicción */}
            {lastPrediction && (
              <div className="bg-white rounded-xl shadow p-6 space-y-4">
                <h3 className="text-base font-semibold text-gray-700">
                  Resultado: <span className="text-gray-500 font-normal">{lastPrediction.filename}</span>
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className={`rounded-xl p-4 text-center ${
                    lastPrediction.is_alertable
                      ? "bg-red-50 border border-red-200"
                      : "bg-green-50 border border-green-200"
                  }`}>
                    <p className="text-3xl mb-2">{lastPrediction.is_alertable ? "🚨" : "✅"}</p>
                    <p className={`text-lg font-bold ${lastPrediction.is_alertable ? "text-red-700" : "text-green-700"}`}>
                      {lastPrediction.is_alertable ? "ALERTABLE" : "NO ALERTABLE"}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      Confianza: {(lastPrediction.binary_confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-center">
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Clase detectada</p>
                    <p className="text-xl font-bold text-blue-800 capitalize">
                      {lastPrediction.prediction.replace(/_/g, " ")}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      Confianza: {(lastPrediction.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                {/* Top K */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm font-medium text-gray-600 mb-3">Top {lastPrediction.top_k.length} clases:</p>
                  <div className="space-y-2">
                    {lastPrediction.top_k.map((item, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <span className="text-xs text-gray-500 w-4">{i + 1}.</span>
                        <span className="text-sm text-gray-700 capitalize w-40">
                          {item.label.replace(/_/g, " ")}
                        </span>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${(item.confidence * 100).toFixed(0)}%` }}
                          />
                        </div>
                        <span className="text-xs text-blue-600 font-medium w-12 text-right">
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        );

      case "eda":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-800">Análisis Exploratorio de Datos (EDA)</h2>
            {edaData ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <EDAChart
                  data={edaData.alertable}
                  title="Distribución de Clases Alertables"
                  color="#ef4444"
                />
                <EDAChart
                  data={edaData.no_alertable}
                  title="Distribución de Clases No Alertables"
                  color="#3b82f6"
                />
              </div>
            ) : (
              <div className="text-gray-400 text-sm">Cargando datos EDA...</div>
            )}
          </div>
        );

      case "metrics":
        return <ModelMetrics />;

      case "wiki":
        return <WikiSection />;

      case "edas":
        return <EDAS />;

      case "preprocesado-y-modelado":
        return <Preprocessing />;

      default:
        return null;
    }
  };

  return (
    <div className="flex min-h-screen bg-gray-100">
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />
      <main className="flex-1 p-6 overflow-y-auto">
        {renderSection()}
      </main>
      <MobileView lastPrediction={lastPrediction} isLoading={isLoading} />
    </div>
  );
}
