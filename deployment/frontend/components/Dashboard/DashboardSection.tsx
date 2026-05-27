"use client";

import DashboardCard from "@/components/DashboardCard";
import AudioTable from "@/components/AudioTable";
import { DashboardStats, PredictionResponse } from "@/types";
import styles from "./DashboardSection.module.css";
interface Props {
    stats: DashboardStats | null;
    predictions: PredictionResponse[];
}

export default function DashboardSection({ stats, predictions }: Props) {
    return (
        <div className="space-y-6">
            {stats ? (
                <div className={[styles["basic__metrics__container"], "grid grid-cols-2 md:grid-cols-4 box black-box gap-4"].join(" ")}>
                    <DashboardCard
                        box_class="box box-blue"
                        title="Total Audios"
                        value={stats.total_files.toLocaleString()}
                        subtitle="Dataset de entrenamiento"
                        color="white"
                        icon="/assets/icons/SVG/nota.svg"

                    />
                    <DashboardCard
                        box_class="box box-blue"
                        title="Clases"
                        value={stats.classes}
                        subtitle="Categorías de sonido"
                        color="purple"
                        icon="/assets/icons/SVG/clases.svg"
                    />
                    <DashboardCard
                        box_class="box box-blue"
                        title="Alertables"
                        value={stats.alertable_count.toLocaleString()}
                        subtitle="Sonidos de alerta"
                        color="red"
                        icon="/assets/icons/SVG/alertable.svg"
                    />
                    <DashboardCard
                        box_class="box box-blue"
                        title="Accuracy"
                        subtitle="Modelo binario"
                        value={`${(stats.model_accuracy * 100).toFixed(1)}%`}
                        currentValue={stats.model_accuracy * 100}
                        maxValue={100}
                        icon="/assets/icons/SVG/presision.svg"
                    />
                </div>
            ) : (
                <div className="text-gray-400 dark:text-gray-500 text-sm">
                    Cargando estadísticas...
                </div>
            )}

            <div className="box black-box rounded-xl shadow p-5">
                <h3 className="text-base font-semibold text-gray-700 dark:text-gray-200 mb-4">
                    Historial de predicciones
                </h3>
                <AudioTable predictions={predictions} />
            </div>
        </div>
    );
}