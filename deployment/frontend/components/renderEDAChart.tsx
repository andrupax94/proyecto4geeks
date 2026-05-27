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
    | "sample_rate";

interface Props {
    activeChart: ActiveChart;
    edaData: EDAData | null;
}

export function renderEDAChart({ activeChart, edaData }: Props) {
    if (!edaData) {
        return (
            <div className="text-gray-400 dark:text-gray-500 text-sm p-10 text-center">
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
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-5">
                
                <div className="mb-6 text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
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

        default:
            return null;
    }
}