"use client";

import { useState } from "react";
import Sidebar from "@/components/sidebar/Sidebar";
import Wallpaper from "@/components/Wallpaper/Wallpaper";
import HomeContent from "@/components/HomeContent";
import { useDashboardData } from "@/hooks/useDashboardData";
import { useAudioPrediction } from "@/hooks/useAudioPrediction";
import { useMobileVisibility } from "@/hooks/useMobileVisibility";
import MobileView from "@/components/MobileView";
type ActiveSection =
  | "dashboard"
  | "predict"
  | "eda"
  | "metrics"
  | "wiki"
  | "edas"
  | "preprocesado-y-modelado";

type ActiveChart =
  | "alertable"
  | "no_alertable"
  | "source"
  | "format"
  | "duration"
  | "sample_rate";

export default function Home() {
  const [activeSection, setActiveSection] = useState<ActiveSection>("dashboard");
  const [activeChart, setActiveChart] = useState<ActiveChart>("alertable");
  const [dragOver, setDragOver] = useState(false);

  const { stats, edaData } = useDashboardData();
  const {
    predictions,
    lastPrediction,
    isLoading,
    error,
    fileInputRef,
    handleFile,
    handleFileChange,
  } = useAudioPrediction();

  const isMobileVisible = useMobileVisibility();

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const handleSectionChange = (section: string) => {
    setActiveSection(section as ActiveSection);
  };

  return (
    <div className="flex min-h-screen">
      <div className="logo"></div>
      <Sidebar activeSection={activeSection} onSectionChange={handleSectionChange} />
      <Wallpaper />

      <HomeContent
        activeSection={activeSection}
        activeChart={activeChart}
        setActiveChart={setActiveChart}
        stats={stats}
        edaData={edaData}
        predictions={predictions}
        lastPrediction={lastPrediction}
        isLoading={isLoading}
        error={error}
        fileInputRef={fileInputRef}
        handleFileChange={handleFileChange}
        handleDrop={handleDrop}
        dragOver={dragOver}
        setDragOver={setDragOver}
      />
      {isMobileVisible && (
        <MobileView lastPrediction={lastPrediction} isLoading={isLoading} />
      )}
    </div>

  );
}