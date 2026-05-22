"use client";

const wikiContent = [
  {
    title: "¿Qué es MIVIA?",
    content:
      "MIVIA (Monitoreo Inteligente de Vigilancia con Inteligencia Artificial) es un sistema de detección y clasificación de sonidos de alerta en tiempo real. Utiliza modelos de deep learning para identificar si un audio contiene sonidos potencialmente peligrosos o alertables.",
  },
  {
    title: "Pipeline de Inferencia Híbrida",
    content:
      "El sistema opera en dos etapas: primero, un modelo binario determina si el audio es 'alertable' o 'no alertable'. Luego, un segundo modelo especializado clasifica el audio dentro de su categoría específica (por ejemplo: sirena, disparo, llanto, etc.).",
  },
  {
    title: "Clases Alertables",
    content:
      "Las clases alertables son: traffic (tráfico), alert_sirem (sirena de alerta), glass_metal (vidrio/metal), gun_shot (disparo), crying (llanto), dog (perro), fight (pelea), construction_noise (ruido de construcción), car_crash (accidente), crime (crimen), fire (fuego) y explosion (explosión).",
  },
  {
    title: "Clases No Alertables",
    content:
      "Las clases no alertables incluyen: instrument (instrumento), domestic_activity (actividad doméstica), ambient_noise (ruido ambiental), voice (voz), another_animal (otro animal), weather (clima), breathing (respiración), water (agua), engine (motor), doors (puertas) y machine (máquina).",
  },
  {
    title: "Arquitectura del Modelo",
    content:
      "Se utiliza la arquitectura HybridCNN v3 (ImprovedMFCCCNN), que combina espectrogramas mel y waveform crudo mediante dos backbones CNN paralelos. La fusión se realiza por concatenación de embeddings de 256 dimensiones, seguida de un clasificador con LayerNorm.",
  },
  {
    title: "Preprocesamiento de Audio",
    content:
      "Los audios se procesan a 16,000 Hz de sample rate, con ventanas de 1024 puntos (n_fft), hop length de 160, 128 bandas mel y 13 coeficientes MFCC. Se aplica normalización de pico y opcionalmente aumentación de dominio para audios externos.",
  },
];

export default function WikiSection() {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-gray-800">Wiki del Proyecto</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {wikiContent.map((item) => (
          <div key={item.title} className="bg-white rounded-xl shadow p-5">
            <h3 className="text-base font-semibold text-blue-700 mb-2">{item.title}</h3>
            <p className="text-sm text-gray-600 leading-relaxed">{item.content}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
