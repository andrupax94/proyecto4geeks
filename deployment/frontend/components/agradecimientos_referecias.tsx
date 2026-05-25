"use client";

interface Recursos {
    title: string;
    content: string;
    link?: string;
}

const recursos: Recursos[] = [
    {
        title: "Recursos utilizados",
        content:
            "Enlaces y recursos utilizados durante la investigación para el desarrollo del proyecto.",
    },

];

const referencias = [
    {
        title: "UrbanSound8K (Kaggle)",
        link: "https://www.kaggle.com/datasets/chrisfilo/urbansound8k",
    },
    {
        title: "FSD50K (Zenodo)",
        link: "https://zenodo.org/records/4060432",
    },
    {
        title: "ESC-50 (GitHub)",
        link: "https://github.com/karolpiczak/ESC-50",
    },
    {
        title: "AudioSet (HuggingFace)",
        link: "https://huggingface.co/datasets/agkphysics/AudioSet",
    },
    {
        title: "Gunshot-audio-dataset (Kaggle)",
        link: "https://www.kaggle.com/datasets/huseyngorbani1/gunshot-audio-dataset?select=gunshot-audio-dataset",
    },
    {
        title: "VOICe Dataset (Zenodo)",
        link: "https://zenodo.org/records/3514950",
    },
    {
        title: "Sound Event Detection for Driver Safety (Kaggle)",
        link: "https://www.kaggle.com/datasets/ccastorena/sound-event-detection-for-driver-safety",
    },
    {
        title: "Emergencysound (Kaggle)",
        link: "https://www.kaggle.com/datasets/buraktaci/emergencysound",
    },
    {
        title: "Enhanced audio of accident and crime detection",
        link: "https://www.kaggle.com/datasets/afisarsy/enhanced-audio-of-accident-and-crime-detection",
    },
    {
        title: "Fondo animado",
        link: "https://codepen.io/franky/collections/",
    },                            
];

export default function Agradecimientos_referecias() {
    return (
        <div className="space-y-6">

            {/* CARD SUPERIOR */}
            <div className="bg-white rounded-xl shadow p-5">
                {recursos.map((item) => (
                    <div key={item.title}>
                        <h3 className="text-base font-semibold text-blue-700 mb-2">
                            {item.title}
                        </h3>

                        <p className="text-sm text-gray-600 leading-relaxed">
                            {item.content}
                        </p>
                    </div>
                ))}
            </div>

            {/* GRID DE REFERENCIAS */}    
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {referencias.map((item) => (
                    <div
                        key={item.title} 
                        className="bg-white rounded-xl shadow p-5"
                        >
                            <a
                                href={item.link} 
                                className="text-blue-700 hover:underline" 
                                target="_blank" 
                                rel="noopener noreferrer"
                            >
                                <h3 className="text-base font-semibold text-blue-700 mb-2">
                                    {item.title}
                                </h3>
                            </a>
                    </div>
                ))}
            </div>
        </div>
    );
}
