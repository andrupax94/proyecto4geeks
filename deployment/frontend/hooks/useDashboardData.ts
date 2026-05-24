"use client";

import { useEffect, useState } from "react";
import { getStats, getEDA } from "@/services/api";
import { DashboardStats, EDAData } from "@/types";

export function useDashboardData() {
    const [stats, setStats] = useState<DashboardStats | null>(null);
    const [edaData, setEdaData] = useState<EDAData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let alive = true;

        async function load() {
            try {
                const [statsRes, edaRes] = await Promise.all([getStats(), getEDA()]);
                if (!alive) return;
                setStats(statsRes);
                setEdaData(edaRes);
            } catch (error) {
                console.error("Error cargando dashboard:", error);
            } finally {
                if (alive) setLoading(false);
            }
        }

        load();

        return () => {
            alive = false;
        };
    }, []);

    return { stats, edaData, loading };
}