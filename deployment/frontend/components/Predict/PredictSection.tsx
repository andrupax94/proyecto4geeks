"use client";

import { PredictionResponse } from "@/types";
import { useRef, useState } from "react";

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
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleFileChangeWithAudio = (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = e.target.files?.[0];
    if (file) {
      // Revoke previous URL to avoid memory leaks
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(file));
      setIsPlaying(false);
    }
    handleFileChange(e);
  };

  const handleDropWithAudio = (e: React.DragEvent) => {
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(file));
      setIsPlaying(false);
    }
    handleDrop(e);
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="space-y-6">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDropWithAudio}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${dragOver
          ? "border-blue-500 bg-blue-50 bg-blue-900/20"
          : "border-gray-300 border-gray-600 hover:border-blue-400 hover:bg-gray-50 hover:bg-gray-700/50"
          }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3,.ogg,.flac,.m4a"
          className="hidden"
          onChange={handleFileChangeWithAudio}
        />

        <div className="flex flex-col items-center gap-3">
          <span className="text-5xl">{isLoading ? "⏳" : "🎙️"}</span>

          {isLoading ? (
            <p className="text-blue-600 text-blue-400 font-medium">
              Analizando audio con el modelo...
            </p>
          ) : (
            <>
              <p className="text-gray-600 text-gray-300 font-medium">
                Arrastra un archivo de audio aquí o haz clic para seleccionar
              </p>
              <p className="text-xs text-gray-300">
                Formatos soportados: .wav, .mp3, .ogg, .flac, .m4a
              </p>
            </>
          )}
        </div>
      </div>

      {/* Audio player */}
      {audioUrl && (
        <div
          className="flex items-center gap-3 box black-box rounded-xl px-4 py-3"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={togglePlay}
            className="flex items-center justify-center w-9 h-9 rounded-full bg-blue-500 hover:bg-blue-600 text-white transition-colors flex-shrink-0"
            aria-label={isPlaying ? "Pausar" : "Reproducir"}
          >
            {isPlaying ? (
              // Pause icon
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="w-4 h-4"
              >
                <path
                  fillRule="evenodd"
                  d="M6.75 5.25a.75.75 0 0 1 .75-.75H9a.75.75 0 0 1 .75.75v13.5a.75.75 0 0 1-.75.75H7.5a.75.75 0 0 1-.75-.75V5.25zm7.5 0A.75.75 0 0 1 15 4.5h1.5a.75.75 0 0 1 .75.75v13.5a.75.75 0 0 1-.75.75H15a.75.75 0 0 1-.75-.75V5.25z"
                  clipRule="evenodd"
                />
              </svg>
            ) : (
              // Play icon
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="w-4 h-4 ml-0.5"
              >
                <path
                  fillRule="evenodd"
                  d="M4.5 5.653c0-1.427 1.529-2.33 2.779-1.643l11.54 6.347c1.295.712 1.295 2.573 0 3.286L7.28 19.99c-1.25.687-2.779-.217-2.779-1.643V5.653z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </button>

          <audio
            ref={audioRef}
            src={audioUrl}
            onEnded={() => setIsPlaying(false)}
            className="hidden"
          />

          <div className="flex-1 min-w-0">
            <p className="text-sm text-gray-200 truncate">
              {fileInputRef.current?.files?.[0]?.name ?? "archivo de audio"}
            </p>
            <p className="text-xs text-gray-300">
              {isPlaying ? "Reproduciendo..." : "Listo para reproducir"}
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 bg-red-900/20 border border-red-200 border-red-800 rounded-lg p-4 text-red-700 text-red-400 text-sm">
          ❌ {error}
        </div>
      )}

      {lastPrediction && (
        <div className="box black-box rounded-xl shadow p-6 space-y-4">
          <h3 className="text-base font-semibold text-gray-200 text-gray-200">
            Resultado:{" "}
            <span className="text-gray-300 font-normal">
              {lastPrediction.filename}
            </span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              className={`rounded-xl p-4 text-center ${lastPrediction.binary_confidence < 0.75 && lastPrediction.binary_confidence > 0.25
                ? "bg-yellow-50 bg-yellow-900/20 border border-yellow-200 border-yellow-800"
                : lastPrediction.is_alertable
                  ? "bg-red-50 bg-red-900/20 border border-red-200 border-red-800"
                  : "bg-green-50 bg-green-900/20 border border-green-200 border-green-800"
                }`}
            >
              <p className="text-3xl mb-2">
                {lastPrediction.binary_confidence < 0.75 && lastPrediction.binary_confidence > 0.25
                  ? "❓"
                  : lastPrediction.is_alertable ? "🚨" : "✅"}
              </p>
              <p
                className={`text-lg font-bold ${lastPrediction.binary_confidence < 0.75 && lastPrediction.binary_confidence > 0.25
                  ? "text-yellow-500 text-yellow-400"
                  : lastPrediction.is_alertable
                    ? "text-red-500 text-red-400"
                    : "text-green-500 text-green-400"
                  }`}
              >
                {lastPrediction.binary_confidence < 0.75 && lastPrediction.binary_confidence > 0.25
                  ? "DESCONOCIDO"
                  : lastPrediction.is_alertable ? "ALERTABLE" : "NO ALERTABLE"}
              </p>
              <p className="text-sm text-gray-300 mt-1">
                Confianza: {(lastPrediction.binary_confidence * 100).toFixed(1)}
                %
              </p>
            </div>

            <div className="bg-blue-50 bg-blue-900/20 border border-blue-200 border-blue-800 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-300 uppercase tracking-wide mb-1">
                Clase detectada
              </p>
              <p className="text-xl font-bold text-blue-500 text-blue-400 capitalize">
                {lastPrediction.prediction.replace(/_/g, " ")}
              </p>
              <p className="text-sm text-gray-300 mt-1">
                Confianza: {(lastPrediction.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="bg-gray-50 bg-gray-700 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-100 mb-3">
              Top {lastPrediction.top_k.length} clases:
            </p>

            <div className="space-y-2">
              {lastPrediction.top_k.map((item, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-xs text-gray-300 w-4">
                    {i + 1}.
                  </span>
                  <span className="text-sm text-gray-200 text-gray-300 capitalize w-40">
                    {item.label.replace(/_/g, " ")}
                  </span>

                  <div className="flex-1 bg-gray-200 bg-gray-600 rounded-full h-2">
                    <div
                      className="bg-blue-500 bg-blue-400 h-2 rounded-full"
                      style={{
                        width: `${(item.confidence * 100).toFixed(0)}%`,
                      }}
                    />
                  </div>

                  <span className="text-xs text-blue-600 text-blue-400 font-medium w-12 text-right">
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
