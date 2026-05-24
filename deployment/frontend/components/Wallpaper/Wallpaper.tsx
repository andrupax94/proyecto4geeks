"use client";

import { useEffect, useRef } from "react";
import { WallpaperService } from "@/services/wallpaper.service";
import styles from "./Wallpaper.module.css";

export default function Wallpaper() {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const serviceRef = useRef<WallpaperService | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        serviceRef.current = new WallpaperService();
        serviceRef.current.start(containerRef.current, {
            velocity: 1,
            density: 15000,
            netLineDistance: 200,
            netLineColor: "#929292",
            particleColors: ["#aaa"],
            spawnQuantity: 3,
        });

        return () => {
            serviceRef.current?.stop();
        };
    }, []);

    return (
        <div className={styles.wallpaper} ref={containerRef}>
            <div className={styles.glow + " " + styles.glow1} />
            <div className={styles.glow + " " + styles.glow2} />
            <div className={styles.glow + " " + styles.glow3} />
        </div>
    );
}