"use client";

import React, { useEffect, useState } from "react";
import DashboardCard from "@/components/DashboardCard";
import AudioTable from "@/components/AudioTable";
import { DashboardStats, PredictionResponse, TrainingData } from "@/types";
import { getTrainingHistory } from "@/services/api";
import styles from "./DashboardSection.module.css";

interface Props {
    stats: DashboardStats | null;
    predictions: PredictionResponse[];
}

export default function DashboardSection({ stats, predictions }: Props) {
    const [alertableData, setAlertableData] = useState<TrainingData | null>(null);
    const [humanData, setHumanData] = useState<TrainingData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchRealMetrics = async () => {
            try {
                const [aData, hData] = await Promise.all([
                    getTrainingHistory("alertable", 4),
                    getTrainingHistory("human_label", 7)
                ]);
                setAlertableData(aData);
                setHumanData(hData);
            } catch (error) {
                console.error("Error fetching real metrics:", error);
            } finally {
                setLoading(false);
            }
        };
        fetchRealMetrics();
    }, []);

    // Extraer métricas reales de los checkpoints
    const getFinalMetric = (data: TrainingData | null, metricKey: keyof any) => {
        if (!data || !data.history || !data.history[metricKey as keyof typeof data.history]) return 0;
        const arr = data.history[metricKey as keyof typeof data.history] as number[];
        return arr.length > 0 ? arr[arr.length - 1] : 0;
    };

    const alertableAccuracy = getFinalMetric(alertableData, "val_acc");
    const humanAccuracy = getFinalMetric(humanData, "val_acc");
    const hybridAccuracy = (alertableAccuracy + humanAccuracy) / 2;

    const humanF1 = getFinalMetric(humanData, "f1");
    const humanPrecision = getFinalMetric(humanData, "precision");
    const humanRecall = getFinalMetric(humanData, "recall");

    return (
        <div className="space-y-6">
            {stats ? (
                <>
                    {/* Fila 1: Estadísticas Generales (Basadas en DB + Modelos Reales) */}
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
                            title="Accuracy Real"
                            subtitle="Checkpoint: Alertable"
                            value={`${(alertableAccuracy * 100).toFixed(1)}%`}
                            currentValue={alertableAccuracy * 100}
                            maxValue={100}
                            icon="/assets/icons/SVG/presision.svg"
                        />
                    </div>

                    {/* Fila 2: Métricas Dinámicas del Modelo */}
                    <div className={[styles["basic__metrics__container"], "grid grid-cols-2 md:grid-cols-4 box black-box gap-4"].join(" ")}>
                        <DashboardCard
                            box_class="box box-blue"
                            title="Accuracy Human"
                            value={`${(humanAccuracy * 100).toFixed(1)}%`}
                            subtitle="Checkpoint: Human Label"
                            color="blue"
                            currentValue={humanAccuracy * 100}
                            icon="/assets/icons/SVG/metricas.svg"
                        />
                        <DashboardCard
                            box_class="box box-blue"
                            title="F1-Score Human"
                            value={humanF1.toFixed(4)}
                            subtitle="Métrica balanceada"
                            color="green"
                            icon="/assets/icons/SVG/metricas.svg"
                        />
                        <DashboardCard
                            box_class="box box-blue"
                            title="Accuracy Híbrido"
                            value={`${(hybridAccuracy * 100).toFixed(1)}%`}
                            subtitle="Promedio de modelos"
                            color="yellow"
                            currentValue={hybridAccuracy * 100}
                            icon="/assets/icons/SVG/testeda.svg"
                        />
                        <DashboardCard
                            box_class="box box-blue"
                            title="Precision Human"
                            value={humanPrecision.toFixed(4)}
                            subtitle="Fiabilidad de etiquetas"
                            color="purple"
                            icon="/assets/icons/SVG/metricas.svg"
                        />
                    </div>

                    {/* Fila 3: Desglose Dinámico (Usando datos de Human Label) */}
                    <div className="box black-box rounded-xl shadow p-5">
                        <h3 className="text-base font-semibold text-gray-200 mb-4 flex items-center gap-2">
                            <img src="/assets/icons/SVG/lista.svg" alt="" className="w-5 h-5 invert opacity-80" />
                            Métricas de Validación Final (Modelo Human Label)
                        </h3>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left text-gray-300">
                                <thead className="text-xs uppercase bg-gray-800/50 text-gray-400">
                                    <tr>
                                        <th className="px-4 py-3">Métrica</th>
                                        <th className="px-4 py-3 text-right">Valor Real</th>
                                        <th className="px-4 py-3 text-right">Estado</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-700">
                                    {[
                                        { name: 'Accuracy', val: humanAccuracy },
                                        { name: 'Precision', val: humanPrecision },
                                        { name: 'Recall', val: humanRecall },
                                        { name: 'F1-Score', val: humanF1 },
                                        { name: 'AUC-ROC', val: getFinalMetric(humanData, "auc_roc") },
                                    ].map((m) => (
                                        <tr key={m.name} className="hover:bg-gray-700/30 transition-colors">
                                            <td className="px-4 py-3 font-medium text-gray-100 capitalize">{m.name}</td>
                                            <td className="px-4 py-3 text-right font-semibold text-blue-400">{(m.val * 100).toFixed(2)}%</td>
                                            <td className="px-4 py-3 text-right">
                                                <span className={`px-2 py-1 rounded-md text-[10px] font-bold ${m.val > 0.85 ? 'bg-green-900/40 text-green-400' : 'bg-yellow-900/40 text-yellow-400'}`}>
                                                    {m.val > 0.85 ? 'ÓPTIMO' : 'ESTABLE'}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            ) : (
                <div className="text-gray-400 dark:text-gray-500 text-sm p-10 text-center animate-pulse">
                    Sincronizando métricas con el servidor...
                </div>
            )}

            <div className="box black-box rounded-xl shadow p-5">
                <h3 className="text-base font-semibold text-gray-200 mb-4">
                    Historial de predicciones
                </h3>
                <AudioTable predictions={predictions} />
            </div>
        </div>
    );
}
