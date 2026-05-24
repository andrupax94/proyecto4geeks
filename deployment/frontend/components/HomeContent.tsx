"use client";

import ModelMetrics from "@/components/ModelMetrics";
import WikiSection from "@/components/WikiSection";
import EDAS from "@/components/EDAS";
import Preprocessing from "@/components/Preprocessing";
import EDASelector from "@/components/EDASelector";
import DashboardSection from "@/components/Dashboard/DashboardSection";
import PredictSection from "@/components/Predict/PredictSection";
import { DashboardStats, EDAData, PredictionResponse } from "@/types";
import { renderEDAChart } from "@/components/renderEDAChart";

type ActiveSection =
    | "dashboard"
    | "predict"
    | "eda"
    | "metrics"
    | "wiki"
    | "edas"
    | "preprocesado-y-modelado";

type ActiveChart =
    | "alertable"
    | "no_alertable"
    | "source"
    | "format"
    | "duration"
    | "sample_rate";

interface Props {
    activeSection: ActiveSection;
    setActiveChart: (chart: ActiveChart) => void;
    activeChart: ActiveChart;
    stats: DashboardStats | null;
    edaData: EDAData | null;
    predictions: PredictionResponse[];
    lastPrediction: PredictionResponse | null;
    isLoading: boolean;
    error: string | null;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    handleDrop: (e: React.DragEvent) => void;
    dragOver: boolean;
    setDragOver: (value: boolean) => void;
}
const sectionTitles: Record<ActiveSection, string> = {
    dashboard: "Dashboard",
    predict: "Predicción de audio",
    eda: "Descripción del dataset",
    metrics: "Métricas del modelo",
    wiki: "Wiki del proyecto",
    edas: "Análisis Exploratorio de Datos (EDAs)",
    "preprocesado-y-modelado": "Evaluación de modelos",
};
export default function HomeContent({
    activeSection,
    activeChart,
    setActiveChart,
    stats,
    edaData,
    predictions,
    lastPrediction,
    isLoading,
    error,
    fileInputRef,
    handleFileChange,
    handleDrop,
    dragOver,
    setDragOver,
}: Props) {
    const renderSection = () => {
        switch (activeSection) {
            case "dashboard":
                return <DashboardSection stats={stats} predictions={predictions} />;

            case "predict":
                return (
                    <PredictSection
                        isLoading={isLoading}
                        error={error}
                        lastPrediction={lastPrediction}
                        fileInputRef={fileInputRef}
                        handleFileChange={handleFileChange}
                        handleDrop={handleDrop}
                        dragOver={dragOver}
                        setDragOver={setDragOver}
                    />
                );

            case "eda":
                return (
                    <div className="space-y-6">

                        <EDASelector
                            activeChart={activeChart}
                            onChartChange={(chart: string) =>
                                setActiveChart(chart as ActiveChart)
                            }
                        />
                        {renderEDAChart({ activeChart, edaData })}
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

    return <main className="flex-1 p-6 overflow-y-auto">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-6">
            {sectionTitles[activeSection]}
        </h2>
        {renderSection()}
    </main>;
}