"use client";

import DashboardCard from "@/components/DashboardCard";
import AudioTable from "@/components/AudioTable";
import { DashboardStats, PredictionResponse } from "@/types";

interface Props {
    stats: DashboardStats | null;
    predictions: PredictionResponse[];
}

export default function DashboardSection({ stats, predictions }: Props) {
    return (
        <div className="space-y-6">


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
                <div className="text-gray-400 dark:text-gray-500 text-sm">
                    Cargando estadísticas...
                </div>
            )}

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-5">
                <h3 className="text-base font-semibold text-gray-700 dark:text-gray-200 mb-4">
                    Historial de predicciones
                </h3>
                <AudioTable predictions={predictions} />
            </div>
        </div>
    );
}