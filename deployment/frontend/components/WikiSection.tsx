"use client";

interface WikiItem {
  title: string;
  content: string;
  link?: string;
}

const wikiContent: WikiItem[] = [
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
      "Se utiliza una arquitectura Hibrida CNN Binaria+CNN Multiclase, que combina espectrogramas mel y waveform crudo mediante dos backbones CNN paralelos. La fusión se realiza por concatenación de embeddings, seguida de un clasificador con LayerNorm."
  },
  {
    title: "¿Por qué CNN?",
    content:
      "Se optó por una arquitectura basada en CNN debido principalmente a restricciones de hardware y compatibilidad del entorno de entrenamiento. Durante el desarrollo se trabajó con una GPU AMD RX 7600 XT bajo ROCm (gfx1102), donde no existían kernels oficiales suficientemente estables o completos para ejecutar de forma óptima arquitecturas más complejas como CRNN o ciertos modelos recurrentes híbridos. Además, no se disponía de una GPU NVIDIA con suficientes núcleos CUDA y soporte maduro para entrenamientos avanzados de modelos secuenciales de mayor costo computacional. \n\nLas CNN ofrecieron un equilibrio adecuado entre rendimiento, estabilidad y compatibilidad, especialmente para el procesamiento de espectrogramas mel y señales de audio. También permitieron aprovechar aceleración por GPU en ROCm con operaciones más estables en fp16/AMP, evitando múltiples problemas conocidos relacionados con kernels de pérdidas recurrentes y operaciones incompatibles en arquitecturas AMD recientes."
  },
  {
    title: "Preprocesamiento de audio",
    content:
      "Los audios se procesan a 16.000 Hz de sample rate, con ventanas de 1.024 puntos (n_fft), hop length de 160, 128 bandas mel y 13 coeficientes MFCC. Se aplica normalización de pico y opcionalmente aumentación de dominio para audios externos.",
  },
  {
    title: "Rendimiento del modelo",
    content:
      "En la evaluación global del pipeline híbrido se observa una disminución del rendimiento respecto a las métricas individuales de cada submodelo. Este comportamiento es esperado en arquitecturas jerárquicas multi-etapa, ya que el sistema requiere que ambas decisiones (clasificación binaria y clasificación multiclase) sean correctas simultáneamente para considerar una predicción final acertada. Como consecuencia, los errores se propagan entre etapas y el rendimiento compuesto tiende a ser inferior al de cada modelo por separado.",
  },
  {
    title: "Rendimiento en una app",
    content:
      "Adicionalmente, el comportamiento del sistema depende significativamente del threshold configurado en el clasificador binario. Por ejemplo, un modo orientado a seguridad puede reducir el threshold para maximizar recall y detectar más eventos potencialmente peligrosos, aunque esto incremente falsos positivos y reduzca la métrica global. En contraste, un modo doméstico o conservador puede elevar el threshold para priorizar precisión y disminuir falsas alarmas.",
  },
  {
    title: "Metricas Multiclase",
    content:
      "Ciertas clases multiclase poseen bajo soporte o presentan características acústicas altamente variables, lo que afecta especialmente métricas como F1 macro. Por ello, la evaluación global debe interpretarse considerando el contexto operacional del sistema y no únicamente como una medida absoluta de desempeño.",
  },
  {
    title: "Visión a futuro",
    content:
      "Si alguien quiere continuar con este proyecto, las líneas más naturales serían: ampliar el dataset con más fuentes de audio en otros idiomas o regiones geográficas, explorar modelos como CRNN o arquitecturas de audio más recientes como los transformers de audio (ya que en este proyecto nos limitamos a utilizar una arquitectura CNN debido a restricciones de hardware), o extender el sistema a detección en tiempo real con dispositivos edge como Raspberry Pi o microcontroladores. El pipeline de construcción que documentamos en el notebook 00 está diseñado precisamente para que incorporar una fuente nueva sea sencillo: solo hay que seguir el mismo proceso aplicado a las 11 fuentes originales.",
  },
  // {
  //   title: "Construcción del dataset",
  //   link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/00.1_building_ds.ipynb",
  //   content:
  //     "El objetivo de este notebook es explicar de dónde vienen los datos, cuáles datasets combinamos, cómo los unificamos y qué significa cada columna del dataset final.",
  // },
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
    title: "Emergency Vehicle Siren Sounds",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.10_EDA_Emergency_Vehicle_Siren_Sounds.ipynb",
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

export default function WikiSection() {
  return (
    <div className="space-y-6">

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {wikiContent.map((item) => (
          <div key={item.title} className="box box-black rounded-xl shadow p-5">
            {item.link ? (
              <a href={item.link} className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                <h3 className="text-base font-semibold text-blue-400 mb-2">{item.title}</h3>
              </a>
            ) : (
              <h3 className="text-base font-semibold text-blue-400 mb-2">{item.title}</h3>
            )}
            <p className="text-sm text-gray-300 leading-relaxed">{item.content}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
