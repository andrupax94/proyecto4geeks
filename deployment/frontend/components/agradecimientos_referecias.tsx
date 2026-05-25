"use client";

interface Agradecimientos {
    title: string;
    content: string;
    link?: string;
}

const agradecimientos: Agradecimientos[] = [
    {
        title: "¿Qué es MIVIA?",
        content:
            "MIVIA (Monitoreo inteligente de vigilancia con Inteligencia Artificial) es un sistema de detección y clasificación de sonidos de alerta. Utiliza modelos de deep learning para identificar si un audio contiene sonidos potencialmente peligrosos o alertables.",
    },

];

const referecias = [
    {
        title: "UrbanSound8K",
        link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/01.1_EDA_UrbanSound8K.ipynb",
    },

];

export default function Agradecimientos_referecias() {
    return (
        <div className="space-y-6">

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {agradecimientos.map((item) => (
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
        </div>
    );
}
