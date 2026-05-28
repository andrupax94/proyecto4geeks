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
    const multiclassStats = stats?.multiclass || {
        accuracy: 0.7977,
        f1_macro: 0.7960,
        f1_weighted: 0.7960
    };

    const hybridAccuracy = stats ? (stats.model_accuracy + multiclassStats.accuracy) / 2 : 0;

    return (
        <div className="space-y-6">
            {stats ? (
                <>
                    {/* Fila 1: Estadísticas Generales (Original) */}
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

                    {/* Fila 2: NUEVA - Modelo Multiclase e Híbrido */}
                    <div className={[styles["basic__metrics__container"], "grid grid-cols-2 md:grid-cols-4 box black-box gap-4"].join(" ")}>
                        <DashboardCard
                            box_class="box box-blue"
                            title="Accuracy Multiclase"
                            value={`${(multiclassStats.accuracy * 100).toFixed(1)}%`}
                            subtitle="Modelo multiclase"
                            color="blue"
                            currentValue={multiclassStats.accuracy * 100}
                            icon="/assets/icons/SVG/metricas.svg"
                        />
                        <DashboardCard
                            box_class="box box-blue"
                            title="F1 Macro Multiclase"
                            value={multiclassStats.f1_macro.toFixed(4)}
                            subtitle="Balance global clases"
                            color="green"
                            icon="/assets/icons/SVG/metricas.svg"
                        />
                        <DashboardCard
                            box_class="box box-blue"
                            title="Accuracy Híbrido"
                            value={`${(hybridAccuracy * 100).toFixed(1)}%`}
                            subtitle="Combinación de modelos"
                            color="yellow"
                            currentValue={hybridAccuracy * 100}
                            icon="/assets/icons/SVG/testeda.svg"
                        />
                        <DashboardCard
                            box_class="box box-blue"
                            title="F1 Weighted"
                            value={multiclassStats.f1_weighted.toFixed(4)}
                            subtitle="Multiclase ponderado"
                            color="purple"
                            icon="/assets/icons/SVG/metricas.svg"
                        />
                    </div>

                    {/* Fila 3: NUEVA - Desglose por clase (multiclase) */}
                    <div className="box black-box rounded-xl shadow p-5">
                        <h3 className="text-base font-semibold text-gray-200 mb-4 flex items-center gap-2">
                            <img src="/assets/icons/SVG/lista.svg" alt="" className="w-5 h-5 invert opacity-80" />
                            Métricas por clase (modelo multiclase)
                        </h3>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left text-gray-300">
                                <thead className="text-xs uppercase bg-gray-800/50 text-gray-400">
                                    <tr>
                                        <th className="px-4 py-3">Clase</th>
                                        <th className="px-4 py-3 text-right">Precision</th>
                                        <th className="px-4 py-3 text-right">Recall</th>
                                        <th className="px-4 py-3 text-right">F1-Score</th>
                                        <th className="px-4 py-3 text-right">Support</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-700">
                                    {[
                                        { name: 'car_crash', p: 0.864, r: 0.864, f1: 0.864, s: 176 },
                                        { name: 'construction_noise', p: 0.856, r: 0.812, f1: 0.834, s: 176 },
                                        { name: 'crying', p: 0.766, r: 0.875, f1: 0.817, s: 176 },
                                        { name: 'dog', p: 0.912, r: 0.881, f1: 0.896, s: 176 },
                                        { name: 'fight', p: 0.962, r: 0.852, f1: 0.904, s: 176 },
                                        { name: 'fire', p: 0.806, r: 0.472, f1: 0.595, s: 176 },
                                        { name: 'glass_breaking', p: 0.887, r: 0.847, f1: 0.866, s: 176 },
                                        { name: 'gun_explosion', p: 0.796, r: 0.955, f1: 0.868, s: 176 },
                                        { name: 'siren_alarm', p: 0.662, r: 0.756, f1: 0.706, s: 176 },
                                        { name: 'traffic', p: 0.565, r: 0.665, f1: 0.611, s: 176 },
                                    ].map((clase) => (
                                        <tr key={clase.name} className="hover:bg-gray-700/30 transition-colors">
                                            <td className="px-4 py-3 font-medium text-gray-100 capitalize">{clase.name.replace('_', ' ')}</td>
                                            <td className="px-4 py-3 text-right">{clase.p.toFixed(3)}</td>
                                            <td className="px-4 py-3 text-right">{clase.r.toFixed(3)}</td>
                                            <td className="px-4 py-3 text-right font-semibold text-blue-400">{clase.f1.toFixed(3)}</td>
                                            <td className="px-4 py-3 text-right text-gray-500">{clase.s}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            ) : (
                <div className="text-gray-400 dark:text-gray-500 text-sm">
                    Cargando estadísticas...
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