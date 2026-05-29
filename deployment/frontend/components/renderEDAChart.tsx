"use client";

import AnimatedChart from "@/components/AnimatedChart";
import EDAChart from "@/components/EDAChart";
import DashboardCard from "@/components/DashboardCard";
import DistributionChart from "@/components/DistributionChart";
import { EDAData } from "@/types";

type ActiveChart =
    | "alertable"
    | "no_alertable"
    | "source"
    | "format"
    | "duration"
    | "sample_rate"
    | "class_metrics";

interface Props {
    activeChart: ActiveChart;
    edaData: EDAData | null;
}

export function renderEDAChart({ activeChart, edaData }: Props) {
    if (!edaData) {
        return (
            <div className="text-gray-400 text-gray-500 text-sm p-10 text-center">
                Cargando datos del dataset...
            </div>
        );
    }

    switch (activeChart) {
        case "alertable":
            return (
                <AnimatedChart isVisible>
                    <EDAChart
                        data={edaData.alertable || []}
                        title="Distribución de clases alertables"
                        color="#ef4444"
                    />
                </AnimatedChart>
            );

        case "no_alertable":
            return (
                <AnimatedChart isVisible>
                    <EDAChart
                        data={edaData.no_alertable || []}
                        title="Distribución de clases no alertables"
                        color="#3b82f6"
                    />
                </AnimatedChart>
            );

        case "source":
            return (
                <AnimatedChart isVisible>
                    <DistributionChart
                        data={(edaData.dataset_source_distribution || []).map((item) => ({
                            name: item.source,
                            value: item.count,
                        }))}
                        title="Distribución por fuente de datos"
                        dataKeyName="name"
                        dataKeyValue="value"
                        color="#f59e0b"
                    />
                </AnimatedChart>
            );

        case "format":
            return (
                <AnimatedChart isVisible>
                    <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box">

                        <div className="mb-6 text-sm text-gray-200 leading-relaxed">
                            <p>
                                En el proyecto se trabajó principalmente con archivos <strong>WAV</strong>,
                                un formato de audio sin compresión que conserva mejor la calidad del sonido
                                y es ampliamente utilizado en inteligencia artificial y procesamiento de audio.
                                También se utilizaron archivos <strong>WAVEX</strong>, una variante basada en WAV
                                que incorpora información adicional del audio, y archivos <strong>MP3</strong>,
                                un formato comprimido más ligero empleado principalmente para pruebas del modelo.
                            </p>

                            <p className="mt-3">
                                Como parte del preprocesamiento, todos los audios fueron convertidos de
                                estéreo a mono, ajustados a una frecuencia de muestreo de <strong>16 kHz</strong>
                                y normalizados a fragmentos de <strong>5 segundos</strong>. Los audios largos
                                se dividieron en segmentos y los más cortos fueron rellenados automáticamente
                                para mantener una duración uniforme en todo el dataset.
                            </p>
                        </div>

                        <DistributionChart
                            data={(edaData.audio_format_distribution || []).map((item) => ({
                                name: item.format,
                                value: item.count,
                            }))}
                            title="Distribución por formato de audio"
                            dataKeyName="name"
                            dataKeyValue="value"
                            color="#10b981"
                        />
                    </div>
                </AnimatedChart>
            );

        case "duration":
            return (
                <AnimatedChart isVisible>
                    <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box">
                        <h3 className="text-base font-semibold text-gray-200 mb-6">
                            Estadísticas de Duración de Audios (segundos)
                        </h3>

                        {edaData.duration_stats ? (
                            <div className=" p-4 rounded-xl grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                                <DashboardCard
                                    title="Media"
                                    value={edaData.duration_stats.mean.toFixed(2)}
                                    subtitle="segundos"
                                    color="blue"
                                />
                                <DashboardCard
                                    title="Mediana"
                                    value={edaData.duration_stats.median.toFixed(2)}
                                    subtitle="segundos"
                                    color="blue"
                                />
                                <DashboardCard
                                    title="Máxima"
                                    value={edaData.duration_stats.max.toFixed(2)}
                                    subtitle="segundos"
                                    color="blue"
                                />
                                <DashboardCard
                                    title="Mínima"
                                    value={edaData.duration_stats.min.toFixed(2)}
                                    subtitle="segundos"
                                    color="blue"
                                />
                                <DashboardCard
                                    title="Desv. Est."
                                    value={edaData.duration_stats.std.toFixed(2)}
                                    subtitle="segundos"
                                    color="blue"
                                />
                            </div>
                        ) : (
                            <p className="text-gray-400 text-sm">
                                No hay datos de duración disponibles.
                            </p>
                        )}
                    </div>
                </AnimatedChart>
            );

        case "sample_rate":
            return (
                <AnimatedChart isVisible>
                    <DistributionChart
                        data={(edaData.sample_rate_distribution || []).map((item) => ({
                            name: `${item.rate / 1000} kHz`,
                            value: item.count,
                        }))}
                        title="Distribución por frecuencia de muestreo"
                        dataKeyName="name"
                        dataKeyValue="value"
                        color="#6366f1"
                    />
                </AnimatedChart>
            );

        case "class_metrics":
            return (
                <AnimatedChart isVisible>
                    <div className="box black-box rounded-xl shadow p-6">
                        <h3 className="text-base font-semibold text-gray-200 mb-6 flex items-center gap-2">
                            📈 Métricas Detalladas por Clase (Modelo Multiclase)
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
                                <tfoot className="bg-gray-800/30 font-bold text-gray-100">
                                    <tr>
                                        <td className="px-4 py-3">Accuracy</td>
                                        <td colSpan={2}></td>
                                        <td className="px-4 py-3 text-right text-green-400">0.798</td>
                                        <td className="px-4 py-3 text-right">1760</td>
                                    </tr>
                                    <tr>
                                        <td className="px-4 py-3">Macro Avg</td>
                                        <td className="px-4 py-3 text-right">0.808</td>
                                        <td className="px-4 py-3 text-right">0.798</td>
                                        <td className="px-4 py-3 text-right">0.796</td>
                                        <td className="px-4 py-3 text-right">1760</td>
                                    </tr>
                                </tfoot>
                            </table>
                        </div>
                    </div>
                </AnimatedChart>
            );

        default:
            return null;
    }
}