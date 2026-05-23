"use client";
const wikiContent = [
  {
    title: "¿Qué es MIVIA?",
    content:
      "MIVIA (Monitoreo inteligente de vigilancia con Inteligencia Artificial) es un sistema de detección y clasificación de sonidos de alerta. Utiliza modelos de deep learning para identificar si un audio contiene sonidos potencialmente peligrosos o alertables.",
  },
  {
    title: "Pipeline de inferencia híbrida",
    content:
      "El sistema opera en dos etapas: primero, un modelo binario determina si el audio es 'alertable' o 'no alertable'. Luego, un segundo modelo especializado clasifica el audio dentro de su categoría específica (por ejemplo: sirena, disparo, llanto, etc.).",
  },
  {
    title: "Clases alertables",
    content:
      "Las clases alertables son: traffic (tráfico), alert_sirem (sirena), glass_metal (vidrio/metal), gun_shot (disparo), crying (llanto), dog (perro), fight (pelea), construction_noise (ruido de construcción), car_crash (accidente), crime (crimen), fire (fuego) y explosion (explosión).",
  },
  {
    title: "Clases no alertables",
    content:
      "Las clases no alertables incluyen: instrument (instrumento), domestic_activity (actividad doméstica), ambient_noise (ruido ambiental), voice (voz), another_animal (otro animal), weather (clima), breathing (respiración), water (agua), engine (motor), doors (puertas) y machine (máquina).",
  },
  {
    title: "Arquitectura del modelo",
    content:
      "Se utiliza la arquitectura HybridCNN v3 (ImprovedMFCCCNN), que combina espectrogramas mel y waveform crudo mediante dos backbones CNN paralelos. La fusión se realiza por concatenación de embeddings, seguida de un clasificador con LayerNorm.",
  },
  {
    title: "Preprocesamiento de audio",
    content:
      "Los audios se procesan a 16.000 Hz de sample rate, con ventanas de 1.024 puntos (n_fft), hop length de 160, 128 bandas mel y 13 coeficientes MFCC. Se aplica normalización de pico y opcionalmente aumentación de dominio para audios externos.",
  },
  {
    title: "Construcción del dataset",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/00.1_building_ds.ipynb",
    content:
      "El objetivo de este notebook es explicar de dónde vienen los datos, cuáles datasets combinamos, cómo los unificamos y qué significa cada columna del dataset final.",
  },
];

const EDAs = [
  {
    title: "UrbanSound8K",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.1_EDA_UrbanSound8K.ipynb",
  },
];

export default function WikiSection() {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-gray-800">Wiki del Proyecto</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {wikiContent.map((item) => (
          <div key={item.title} className="bg-white rounded-xl shadow p-5">
            {item.link ? (
              <a href={item.link} className="text-blue-700 hover:underline" target="_blank" rel="noopener noreferrer">
                <h3 className="text-base font-semibold text-blue-700 mb-2">{item.title}</h3>
              </a>
            ) : (
              <h3 className="text-base font-semibold text-blue-700 mb-2">{item.title}</h3>
            )}
            <p className="text-sm text-gray-600 leading-relaxed">{item.content}</p>
          </div>
        ))}
      </div>
      <div className="bg-white rounded-xl shadow p-5">
        <h3 className="text-xl font-bold text-blue-700">EDAs (Análisis Exploratorio de Datos)</h3>
        <p className="text-sm text-gray-600 leading-relaxed">En este apartado se presentan los análisis exploratorios de los datos utilizados en el proyecto.</p>
        <ul className="list-disc list-inside mt-3 space-y-2">
          {EDAs.map((eda) => (
            <li key={eda.title}>
              <a href={eda.link} className="text-blue-700 hover:underline" target="_blank" rel="noopener noreferrer">
                {eda.title}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
