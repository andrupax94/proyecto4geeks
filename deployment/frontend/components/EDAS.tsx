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
      "Se utiliza la arquitectura HybridCNN v3 (ImprovedMFCCCNN), que combina espectrogramas mel y waveform crudo mediante dos backbones CNN paralelos. La fusión se realiza por concatenación de embeddings, seguida de un clasificador con LayerNorm.\n\nAunque la evaluación global del sistema (clasificación binaria y multiclase combinadas) presenta una ligera disminución en el rendimiento, la arquitectura híbrida obtiene mejores resultados generales gracias a su capacidad para aprovechar simultáneamente características espectrales y temporales del audio.",
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
  {
    title: "FSD50K",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.2_EDA_FSD50K.ipynb",
  },
  {
    title: "ESC-50",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.3_EDA_ESC_50.ipynb",
  },
  {
    title: "AudioSet",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.4_EDA_AudioSet.ipynb",
  },
  {
    title: "Gunshot-audio-dataset",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.5_EDA_gunshot_audio_dataset.ipynb",
  },
  {
    title: "VOICe Dataset",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.6_EDA_VOICe%20(corregido).ipynb",
  },
  {
    title: "Sound Event Detection for Driver Safety",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.7_EDA_Sound_Event_Detection_for_Driver_Safety.ipynb",
  },
  {
    title: "Emergencysound",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.8_EDA_emergencysound.ipynb",
  },
  {
    title: "Enhanced audio of accident and crime detection",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.9_EDA_Enhanced_audio_of_accident_and_crime_detection.ipynb",
  },
  {
    title: "Vídeos varios (YouTube)",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.11_EDA_youtube.ipynb",
  },
  {
    title: "Edge-collected-gunshot dataset (CSV)",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.12%20EDA_Edge_collected_gunshot.ipynb",
  },
];

const finalDatasetEDAs = [
  {
    title: "EDA del dataset final",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/02.1_EDA_DatasetFinal.ipynb"
  },
  {
    title: "EDA acústico",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/02.2_EDA_ac%C3%BAstico.ipynb"
  },
  {
    title: "EDA sampling alternativo",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/02.3_EDA_sampling_alternativo.ipynb"
  },
];

export default function EDAS() {
  return (
    <div className="space-y-6">

      <div className="bg-white rounded-xl shadow p-5">
        <h2 className="text-base font-semibold text-blue-700 mb-2">Construcción del dataset</h2>
        <p className="text-sm text-gray-600 leading-relaxed">El objetivo de este notebook es explicar de dónde vienen los datos, cuáles datasets combinamos, cómo los unificamos y qué significa cada columna del dataset final.</p>
        <a href="https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/00.1_building_ds.ipynb" className="text-blue-700 hover:underline" target="_blank" rel="noopener noreferrer">
          Notebook de construcción del dataset
        </a>
      </div>
      <div className="bg-white rounded-xl shadow p-5">
        <h3 className="text-base font-semibold text-blue-700 mb-2">Recopilación de los EDAs</h3>
        <p className="text-sm text-gray-600 leading-relaxed">En este apartado se presentan los análisis exploratorios de los datos utilizados en el proyecto.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <ul className="list-disc list-inside mt-3 space-y-2">
            {EDAs.slice(0, EDAs.length / 2).map((eda) => (
              <li key={eda.title}>
                <a href={eda.link} className="text-blue-700 hover:underline" target="_blank" rel="noopener noreferrer">
                  {eda.title}
                </a>
              </li>
            ))}
          </ul>
          <ul className="list-disc list-inside mt-3 space-y-2">
            {EDAs.slice(EDAs.length / 2).map((eda) => (
              <li key={eda.title}>
                <a href={eda.link} className="text-blue-700 hover:underline" target="_blank" rel="noopener noreferrer">
                  {eda.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div className="bg-white rounded-xl shadow p-5">
        <h3 className="text-base font-semibold text-blue-700 mb-2">EDA del dataset final</h3>
        <p className="text-sm text-gray-600 leading-relaxed">Análisis exploratorio del dataset final construido.</p>
        <ul className="list-disc list-inside mt-3 space-y-2">
          {finalDatasetEDAs.map((eda) => (
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
