"use client";

import { useEffect, useState } from "react";
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
} from "recharts";
import AnimatedChart from "@/components/AnimatedChart";
import DashboardCard from "@/components/DashboardCard";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface TrainingHistory {
    epoch: number[];
    train_loss: number[];
    train_acc: number[];
    val_loss: number[];
    val_acc: number[];
    precision: number[];
    recall: number[];
    f1: number[];
    lr: number[];
    epoch_time: number[];
    images_per_sec: number[];
}

interface TrainingData {
    checkpoint_epoch: number;
    best_acc: number;
    best_epoch: number;
    history: TrainingHistory;
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/** Convierte los arrays paralelos del history en un array de objetos por epoch */
function buildRows(history: TrainingHistory) {
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
}

// ─────────────────────────────────────────────
// Custom Tooltip
// ─────────────────────────────────────────────

const CustomTooltip = ({
    active,
    payload,
    label,
    unit = "",
}: {
    active?: boolean;
    payload?: { name: string; value: number; color: string }[];
    label?: number;
    unit?: string;
}) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl text-xs">
            <p className="text-gray-400 mb-1 font-medium">Epoch {label}</p>
            {payload.map((p) => (
                <p key={p.name} style={{ color: p.color }} className="font-semibold">
                    {p.name}: {p.value}
                    {unit}
                </p>
            ))}
        </div>
    );
};

// ─────────────────────────────────────────────
// Sub-charts
// ─────────────────────────────────────────────

function LossChart({
    rows,
    bestEpoch,
}: {
    rows: ReturnType<typeof buildRows>;
    bestEpoch: number;
}) {
    return (
        <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box">
            <h3 className="text-base font-semibold text-gray-200 mb-4">
                📉 Loss — Train vs Validación
            </h3>
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                        dataKey="epoch"
                        tick={{ fill: "#9ca3af", fontSize: 11 }}
                        label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: "#6b7280", fontSize: 11 }}
                    />
                    <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        wrapperStyle={{ fontSize: 12, color: "#d1d5db" }}
                        iconType="circle"
                        iconSize={8}
                    />
                    <ReferenceLine
                        x={bestEpoch}
                        stroke="#facc15"
                        strokeDasharray="4 4"
                        label={{ value: "Best", fill: "#facc15", fontSize: 10, position: "top" }}
                    />
                    <Line
                        type="monotone"
                        dataKey="train_loss"
                        name="Train Loss"
                        stroke="#f87171"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                    />
                    <Line
                        type="monotone"
                        dataKey="val_loss"
                        name="Val Loss"
                        stroke="#60a5fa"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

function AccuracyChart({
    rows,
    bestEpoch,
}: {
    rows: ReturnType<typeof buildRows>;
    bestEpoch: number;
}) {
    return (
        <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box">
            <h3 className="text-base font-semibold text-gray-200 mb-4">
                🎯 Accuracy — Train vs Validación
            </h3>
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <YAxis
                        tick={{ fill: "#9ca3af", fontSize: 11 }}
                        domain={[0, 100]}
                        tickFormatter={(v) => `${v}%`}
                    />
                    <Tooltip content={<CustomTooltip unit="%" />} />
                    <Legend
                        wrapperStyle={{ fontSize: 12, color: "#d1d5db" }}
                        iconType="circle"
                        iconSize={8}
                    />
                    <ReferenceLine
                        x={bestEpoch}
                        stroke="#facc15"
                        strokeDasharray="4 4"
                        label={{ value: "Best", fill: "#facc15", fontSize: 10, position: "top" }}
                    />
                    <Line
                        type="monotone"
                        dataKey="train_acc"
                        name="Train Acc"
                        stroke="#f87171"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                    />
                    <Line
                        type="monotone"
                        dataKey="val_acc"
                        name="Val Acc"
                        stroke="#34d399"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

function MetricsChart({ rows, bestEpoch }: { rows: ReturnType<typeof buildRows>; bestEpoch: number }) {
    return (
        <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box">
            <h3 className="text-base font-semibold text-gray-200 mb-4">
                📊 Precision · Recall · F1
            </h3>
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <YAxis
                        tick={{ fill: "#9ca3af", fontSize: 11 }}
                        domain={[0, 100]}
                        tickFormatter={(v) => `${v}%`}
                    />
                    <Tooltip content={<CustomTooltip unit="%" />} />
                    <Legend
                        wrapperStyle={{ fontSize: 12, color: "#d1d5db" }}
                        iconType="circle"
                        iconSize={8}
                    />
                    <ReferenceLine
                        x={bestEpoch}
                        stroke="#facc15"
                        strokeDasharray="4 4"
                        label={{ value: "Best", fill: "#facc15", fontSize: 10, position: "top" }}
                    />
                    <Line type="monotone" dataKey="precision" name="Precision" stroke="#a78bfa" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    <Line type="monotone" dataKey="recall" name="Recall" stroke="#fb923c" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    <Line type="monotone" dataKey="f1" name="F1 Score" stroke="#38bdf8" strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

function LRChart({ rows }: { rows: ReturnType<typeof buildRows> }) {
    return (
        <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box">
            <h3 className="text-base font-semibold text-gray-200 mb-4">
                📈 Learning Rate Schedule
            </h3>
            <ResponsiveContainer width="100%" height={200}>
                <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <YAxis
                        tick={{ fill: "#9ca3af", fontSize: 11 }}
                        tickFormatter={(v) =>
                            v < 0.0001 ? v.toExponential(1) : v.toFixed(5)
                        }
                        width={60}
                    />
                    <Tooltip
                        content={({ active, payload, label }) => {
                            if (!active || !payload?.length) return null;
                            return (
                                <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 shadow-xl text-xs">
                                    <p className="text-gray-400 mb-1">Epoch {label}</p>
                                    <p className="text-yellow-400 font-semibold">
                                        LR: {Number(payload[0].value).toExponential(3)}
                                    </p>
                                </div>
                            );
                        }}
                    />
                    <Line
                        type="monotone"
                        dataKey="lr"
                        name="LR"
                        stroke="#facc15"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

function SpeedChart({ rows }: { rows: ReturnType<typeof buildRows> }) {
    return (
        <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box">
            <h3 className="text-base font-semibold text-gray-200 mb-4">
                ⚡ Velocidad — Imágenes / segundo
            </h3>
            <ResponsiveContainer width="100%" height={200}>
                <LineChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="epoch" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip unit=" img/s" />} />
                    <Line
                        type="monotone"
                        dataKey="images_per_sec"
                        name="img/s"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

// ─────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────

interface TrainingHistoryChartsProps {
    /** Si ya tienes los datos cargados, pásalos directamente */
    data?: TrainingData | null;
    /** O pásale la URL del endpoint y el componente hace el fetch */
    apiUrl?: string;
    targetType?: string;
    version?: number;
}

export default function TrainingHistoryCharts({
    data: externalData,
    apiUrl = "/api/training/history",
    targetType = "alertable",
    version = 4,
}: TrainingHistoryChartsProps) {
    const [data, setData] = useState<TrainingData | null>(externalData ?? null);
    const [loading, setLoading] = useState(!externalData);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (externalData) return;
        setLoading(true);
        fetch(`${apiUrl}?target_type=${targetType}&version=${version}`)
            .then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then((d: TrainingData) => setData(d))
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false));
    }, [apiUrl, targetType, version, externalData]);

    if (loading) {
        return (
            <div className="text-gray-400 text-sm p-10 text-center">
                Cargando historial de entrenamiento...
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="text-red-400 text-sm p-10 text-center">
                {error ?? "No hay datos disponibles."}
            </div>
        );
    }

    const rows = buildRows(data.history);
    const lastRow = rows[rows.length - 1];
    const totalTime = data.history.epoch_time.reduce((a, b) => a + b, 0);

    return (
        <div className="flex flex-col gap-6">

            {/* ── KPI cards ── */}
            <AnimatedChart isVisible>
                <div className="box black-box p-5 shadow">
                    <h3 className="text-base font-semibold text-gray-200 mb-6">
                        Resumen del Entrenamiento — Checkpoint epoch {data.checkpoint_epoch}
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                        <DashboardCard
                            title="Best Val Acc"
                            value={`${(data.best_acc * 100).toFixed(2)}%`}
                            subtitle={`epoch ${data.best_epoch}`}
                            color="blue"
                        />
                        <DashboardCard
                            title="Último F1"
                            value={`${lastRow?.f1 ?? "—"}%`}
                            subtitle="weighted"
                            color="blue"
                        />
                        <DashboardCard
                            title="Última Precision"
                            value={`${lastRow?.precision ?? "—"}%`}
                            subtitle="weighted"
                            color="blue"
                        />
                        <DashboardCard
                            title="Último Recall"
                            value={`${lastRow?.recall ?? "—"}%`}
                            subtitle="weighted"
                            color="blue"
                        />
                        <DashboardCard
                            title="Epochs"
                            value={String(data.checkpoint_epoch)}
                            subtitle="completadas"
                            color="blue"
                        />
                        <DashboardCard
                            title="Tiempo total"
                            value={`${(totalTime / 60).toFixed(1)} min`}
                            subtitle="entrenamiento"
                            color="blue"
                        />
                    </div>
                </div>
            </AnimatedChart>

            {/* ── Loss + Accuracy ── */}
            <AnimatedChart isVisible>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <LossChart rows={rows} bestEpoch={data.best_epoch} />
                    <AccuracyChart rows={rows} bestEpoch={data.best_epoch} />
                </div>
            </AnimatedChart>

            {/* ── Precision / Recall / F1 ── */}
            <AnimatedChart isVisible>
                <MetricsChart rows={rows} bestEpoch={data.best_epoch} />
            </AnimatedChart>

            {/* ── LR + Speed ── */}
            <AnimatedChart isVisible>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <LRChart rows={rows} />
                    <SpeedChart rows={rows} />
                </div>
            </AnimatedChart>
        </div>
    );
}
