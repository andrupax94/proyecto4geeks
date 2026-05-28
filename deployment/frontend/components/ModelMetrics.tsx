"use client";

import React, { useEffect, useState } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ReferenceLine,
    RadarChart,
    Radar,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
} from "recharts";
import { getTrainingHistory } from "@/services/api";
import { TrainingData, TrainingHistory } from "@/types";

// ─────────────────────────────────────────────
// Tipos y Constantes
// ─────────────────────────────────────────────

type ModelType = "alertable" | "human_label";

interface ComparisonData {
    metric: string;
    alertable: number;
    human_label: number;
}

// ─────────────────────────────────────────────
// Componentes Modulares
// ─────────────────────────────────────────────

const CustomTooltip = ({
    active,
    payload,
    label,
    unit = "",
}: {
    active?: boolean;
    payload?: any[];
    label?: any;
    unit?: string;
}) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl text-xs">
            <p className="text-gray-400 mb-1 font-medium">{typeof label === 'number' ? `Epoch ${label}` : label}</p>
            {payload.map((p, i) => (
                <p key={i} style={{ color: p.color }} className="font-semibold">
                    {p.name}: {p.value}
                    {unit}
                </p>
            ))}
        </div>
    );
};

const ChartContainer = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box rounded-xl">
        <h3 className="text-base font-semibold text-gray-200 mb-4">{title}</h3>
        {children}
    </div>
);

const HistoryCharts = ({ rows, bestEpoch }: { rows: any[]; bestEpoch: number }) => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartContainer title="📉 Loss — Train vs Validación">
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#d1d5db" }} iconType="circle" iconSize={8} />
                    <ReferenceLine x={bestEpoch} stroke="#facc15" strokeDasharray="4 4" label={{ value: "Best", fill: "#facc15", fontSize: 10, position: "top" }} />
                    <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#f87171" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#60a5fa" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
            </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer title="🎯 Accuracy — Train vs Validación">
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                    <Tooltip content={<CustomTooltip unit="%" />} />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#d1d5db" }} iconType="circle" iconSize={8} />
                    <ReferenceLine x={bestEpoch} stroke="#facc15" strokeDasharray="4 4" label={{ value: "Best", fill: "#facc15", fontSize: 10, position: "top" }} />
                    <Line type="monotone" dataKey="train_acc" name="Train Acc" stroke="#f87171" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    <Line type="monotone" dataKey="val_acc" name="Val Acc" stroke="#34d399" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
            </ResponsiveContainer>
        </ChartContainer>

        <ChartContainer title="📊 Precision · Recall · F1">
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                    <Tooltip content={<CustomTooltip unit="%" />} />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#d1d5db" }} iconType="circle" iconSize={8} />
                    <ReferenceLine x={bestEpoch} stroke="#facc15" strokeDasharray="4 4" label={{ value: "Best", fill: "#facc15", fontSize: 10, position: "top" }} />
                    <Line type="monotone" dataKey="precision" name="Precision" stroke="#a78bfa" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    <Line type="monotone" dataKey="recall" name="Recall" stroke="#fb923c" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    <Line type="monotone" dataKey="f1" name="F1 Score" stroke="#38bdf8" strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
            </ResponsiveContainer>
        </ChartContainer>

        <div className="grid grid-cols-1 gap-6">
            <ChartContainer title="📈 Learning Rate Schedule">
                <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                        <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} tickFormatter={(v) => v < 0.0001 ? v.toExponential(1) : v.toFixed(5)} width={60} />
                        <Tooltip content={({ active, payload, label }) => {
                            if (!active || !payload?.length) return null;
                            return (
                                <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl text-xs">
                                    <p className="text-gray-400 mb-1">Epoch {label}</p>
                                    <p className="text-yellow-400 font-semibold">LR: {Number(payload[0].value).toExponential(3)}</p>
                                </div>
                            );
                        }} />
                        <Line type="monotone" dataKey="lr" name="LR" stroke="#facc15" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    </LineChart>
                </ResponsiveContainer>
            </ChartContainer>
            <ChartContainer title="⚡ Velocidad — Imágenes / segundo">
                <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                        <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
                        <Tooltip content={<CustomTooltip unit=" img/s" />} />
                        <Line type="monotone" dataKey="images_per_sec" name="img/s" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    </LineChart>
                </ResponsiveContainer>
            </ChartContainer>
        </div>
    </div>
);

const ComparisonMetrics = ({ comparisonData, alertableInfo, humanInfo }: { comparisonData: ComparisonData[], alertableInfo: any, humanInfo: any }) => (
    <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartContainer title="Rendimiento Comparativo (Radar)">
                <ResponsiveContainer width="100%" height={350}>
                    <RadarChart data={comparisonData}>
                        <PolarGrid stroke="#374151" />
                        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: "#9ca3af" }} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10, fill: "#4b5563" }} />
                        <Radar name="Alertable" dataKey="alertable" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                        <Radar name="Human Label" dataKey="human_label" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                        <Tooltip content={<CustomTooltip unit="%" />} />
                        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 20 }} />
                    </RadarChart>
                </ResponsiveContainer>
            </ChartContainer>

            <ChartContainer title="Resumen de Métricas Finales">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="text-gray-400 border-b border-gray-800">
                            <th className="px-4 py-2 text-left">Métrica</th>
                            <th className="px-4 py-2 text-right text-blue-400">Alertable</th>
                            <th className="px-4 py-2 text-right text-green-400">Human</th>
                        </tr>
                    </thead>
                    <tbody>
                        {comparisonData.map((m) => (
                            <tr key={m.metric} className="border-b border-gray-800/50 hover:bg-gray-800/20 transition-colors">
                                <td className="px-4 py-3 font-medium text-gray-300">{m.metric}</td>
                                <td className="px-4 py-3 text-right text-blue-400 font-bold">{m.alertable.toFixed(1)}%</td>
                                <td className="px-4 py-3 text-right text-green-400 font-bold">{m.human_label.toFixed(1)}%</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </ChartContainer>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ChartContainer title="Detalles: Modelo Alertable">
                <div className="space-y-3">
                    <DetailRow label="Archivo" value={alertableInfo?.model_name || "N/A"} />
                    <DetailRow label="Mejor Accuracy" value={`${(alertableInfo?.best_acc * 100 || 0).toFixed(2)}%`} />
                    <DetailRow label="Mejor Epoch" value={alertableInfo?.best_epoch || "N/A"} />
                    <DetailRow label="Versión" value={`V${alertableInfo?.version || "4"}`} />
                </div>
            </ChartContainer>
            <ChartContainer title="Detalles: Modelo Human Label">
                <div className="space-y-3">
                    <DetailRow label="Archivo" value={humanInfo?.model_name || "N/A"} />
                    <DetailRow label="Mejor Accuracy" value={`${(humanInfo?.best_acc * 100 || 0).toFixed(2)}%`} />
                    <DetailRow label="Mejor Epoch" value={humanInfo?.best_epoch || "N/A"} />
                    <DetailRow label="Versión" value={`V${humanInfo?.version || "4"}`} />
                </div>
            </ChartContainer>
        </div>
    </div>
);

const DetailRow = ({ label, value }: { label: string; value: string | number }) => (
    <div className="flex justify-between items-center py-2 border-b border-gray-800/50">
        <span className="text-sm text-gray-400">{label}</span>
        <span className="text-sm font-medium text-gray-200">{value}</span>
    </div>
);

// ─────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────

export default function ModelMetrics() {
    const [modelType, setModelType] = useState<ModelType>("alertable");
    const [data, setData] = useState<TrainingData | null>(null);
    const [alertableData, setAlertableData] = useState<TrainingData | null>(null);
    const [humanData, setHumanData] = useState<TrainingData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showHistory, setShowHistory] = useState(true);

    useEffect(() => {
        const fetchAllData = async () => {
            setLoading(true);
            setError(null);
            try {
                const [aData, hData] = await Promise.all([
                    getTrainingHistory("alertable", 4),
                    getTrainingHistory("human_label", 6)
                ]);
                setAlertableData(aData);
                setHumanData(hData);
                setData(modelType === "alertable" ? aData : hData);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchAllData();
    }, []);

    useEffect(() => {
        if (alertableData && humanData) {
            setData(modelType === "alertable" ? alertableData : humanData);
        }
    }, [modelType, alertableData, humanData]);

    const buildRows = (history: TrainingHistory) => {
        return history.epoch.map((ep, i) => ({
            epoch: ep,
            train_loss: +(history.train_loss[i] ?? 0).toFixed(4),
            val_loss: +(history.val_loss[i] ?? 0).toFixed(4),
            train_acc: +(history.train_acc[i] * 100).toFixed(2),
            val_acc: +(history.val_acc[i] * 100).toFixed(2),
            precision: +(history.precision[i] * 100).toFixed(2),
            recall: +(history.recall[i] * 100).toFixed(2),
            f1: +(history.f1[i] * 100).toFixed(2),
            lr: history.lr[i],
            epoch_time: +(history.epoch_time[i] ?? 0).toFixed(1),
            images_per_sec: +(history.images_per_sec[i] ?? 0).toFixed(0),
        }));
    };

    const getComparisonData = (): ComparisonData[] => {
        if (!alertableData || !humanData) return [];

        const aHist = alertableData.history;
        const hHist = humanData.history;

        // Obtenemos el último valor de cada métrica
        const getLast = (arr: number[]) => (arr && arr.length > 0 ? arr[arr.length - 1] * 100 : 0);

        return [
            { metric: "Accuracy", alertable: getLast(aHist.val_acc), human_label: getLast(hHist.val_acc) },
            { metric: "Precision", alertable: getLast(aHist.precision), human_label: getLast(hHist.precision) },
            { metric: "Recall", alertable: getLast(aHist.recall), human_label: getLast(hHist.recall) },
            { metric: "F1-Score", alertable: getLast(aHist.f1), human_label: getLast(hHist.f1) },
            { metric: "AUC-ROC", alertable: getLast(aHist.auc_roc || []), human_label: getLast(hHist.auc_roc || []) },
        ];
    };

    return (
        <div className="space-y-6">
            {/* Header with Selector */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-gray-900/50 p-4 rounded-xl border border-gray-800">
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowHistory(true)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${showHistory ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"}`}
                    >
                        Historial de Entrenamiento
                    </button>
                    <button
                        onClick={() => setShowHistory(false)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${!showHistory ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"}`}
                    >
                        Comparativa de Modelos
                    </button>
                </div>

                {showHistory && (
                    <div className="flex items-center bg-black rounded-lg p-1 border border-gray-700">
                        <button
                            onClick={() => setModelType("alertable")}
                            className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${modelType === "alertable" ? "bg-gray-800 text-blue-400 shadow-sm" : "text-gray-500 hover:text-gray-300"}`}
                        >
                            ALERTABLE
                        </button>
                        <button
                            onClick={() => setModelType("human_label")}
                            className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${modelType === "human_label" ? "bg-gray-800 text-blue-400 shadow-sm" : "text-gray-500 hover:text-gray-300"}`}
                        >
                            HUMAN LABEL
                        </button>
                    </div>
                )}
            </div>

            {loading ? (
                <div className="flex flex-col items-center justify-center p-20 space-y-4">
                    <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                    <p className="text-gray-400 animate-pulse">Sincronizando datos de los modelos...</p>
                </div>
            ) : error ? (
                <div className="bg-red-900/20 border border-red-900/50 p-6 rounded-xl text-center">
                    <p className="text-red-400 mb-2">❌ Error: {error}</p>
                    <button onClick={() => window.location.reload()} className="text-sm text-blue-400 hover:underline">Reintentar</button>
                </div>
            ) : (
                showHistory ? (
                    data && <HistoryCharts rows={buildRows(data.history)} bestEpoch={data.best_epoch} />
                ) : (
                    <ComparisonMetrics
                        comparisonData={getComparisonData()}
                        alertableInfo={alertableData}
                        humanInfo={humanData}
                    />
                )
            )}
        </div>
    );
}
