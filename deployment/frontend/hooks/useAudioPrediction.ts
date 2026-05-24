"use client";

import { useRef, useState } from "react";
import { predictAudio } from "@/services/api";
import { PredictionResponse } from "@/types";

export function useAudioPrediction() {
    const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
    const [lastPrediction, setLastPrediction] = useState<PredictionResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFile = async (file: File) => {
        if (!file) return;

        setIsLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const result = await predictAudio(formData);
            setPredictions((prev) => [result, ...prev]);
            setLastPrediction(result);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Error desconocido al predecir");
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleFile(file);
    };

    return {
        predictions,
        lastPrediction,
        isLoading,
        error,
        fileInputRef,
        handleFile,
        handleFileChange,
        setError,
    };
}