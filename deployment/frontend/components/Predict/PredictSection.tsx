"use client";

import { PredictionResponse } from "@/types";

interface Props {
    isLoading: boolean;
    error: string | null;
    lastPrediction: PredictionResponse | null;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    handleDrop: (e: React.DragEvent) => void;
    dragOver: boolean;
    setDragOver: (value: boolean) => void;
}

export default function PredictSection({
    isLoading,
    error,
    lastPrediction,
    fileInputRef,
    handleFileChange,
    handleDrop,
    dragOver,
    setDragOver,
}: Props) {
    return (
        <div className="space-y-6">


            <div
                onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${dragOver
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                    : "border-gray-300 dark:border-gray-600 hover:border-blue-400 hover:bg-gray-50 dark:hover:bg-gray-700/50"
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
                        <p className="text-blue-600 dark:text-blue-400 font-medium">
                            Analizando audio con el modelo...
                        </p>
                    ) : (
                        <>
                            <p className="text-gray-600 dark:text-gray-300 font-medium">
                                Arrastra un archivo de audio aquí o haz clic para seleccionar
                            </p>
                            <p className="text-xs text-gray-400 dark:text-gray-500">
                                Formatos soportados: .wav, .mp3, .ogg, .flac, .m4a
                            </p>
                        </>
                    )}
                </div>
            </div>

            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-400 text-sm">
                    ❌ {error}
                </div>
            )}

            {lastPrediction && (
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6 space-y-4">
                    <h3 className="text-base font-semibold text-gray-700 dark:text-gray-200">
                        Resultado:{" "}
                        <span className="text-gray-500 dark:text-gray-400 font-normal">
                            {lastPrediction.filename}
                        </span>
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div
                            className={`rounded-xl p-4 text-center ${lastPrediction.is_alertable
                                ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
                                : "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
                                }`}
                        >
                            <p className="text-3xl mb-2">
                                {lastPrediction.is_alertable ? "🚨" : "✅"}
                            </p>
                            <p
                                className={`text-lg font-bold ${lastPrediction.is_alertable
                                    ? "text-red-700 dark:text-red-400"
                                    : "text-green-700 dark:text-green-400"
                                    }`}
                            >
                                {lastPrediction.is_alertable ? "ALERTABLE" : "NO ALERTABLE"}
                            </p>
                            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                                Confianza: {(lastPrediction.binary_confidence * 100).toFixed(1)}%
                            </p>
                        </div>

                        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4 text-center">
                            <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1">
                                Clase detectada
                            </p>
                            <p className="text-xl font-bold text-blue-800 dark:text-blue-400 capitalize">
                                {lastPrediction.prediction.replace(/_/g, " ")}
                            </p>
                            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                                Confianza: {(lastPrediction.confidence * 100).toFixed(1)}%
                            </p>
                        </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                        <p className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-3">
                            Top {lastPrediction.top_k.length} clases:
                        </p>

                        <div className="space-y-2">
                            {lastPrediction.top_k.map((item, i) => (
                                <div key={i} className="flex items-center gap-3">
                                    <span className="text-xs text-gray-500 dark:text-gray-400 w-4">
                                        {i + 1}.
                                    </span>
                                    <span className="text-sm text-gray-700 dark:text-gray-300 capitalize w-40">
                                        {item.label.replace(/_/g, " ")}
                                    </span>

                                    <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                        <div
                                            className="bg-blue-500 dark:bg-blue-400 h-2 rounded-full"
                                            style={{ width: `${(item.confidence * 100).toFixed(0)}%` }}
                                        />
                                    </div>

                                    <span className="text-xs text-blue-600 dark:text-blue-400 font-medium w-12 text-right">
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
}