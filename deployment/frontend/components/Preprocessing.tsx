"use client";

const preprocessingAndModeling = [
  {
    title: "Preprocesado",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/03.1_preprocessing.ipynb"
  },
  {
    title: "Modelado",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/03.2_modeling.ipynb"
  },
];

const evaluations = [
  {
    title: "Alertable",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/04.1_evaluation_alertable.ipynb"
  },
  {
    title: "Multiclase",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/04.2_evaluation_multiclase.ipynb"
  },
  {
    title: "Híbrida",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/04.3_evaluation_hybrid.ipynb"
  },
  {
    title: "No alertable",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/04.4_evaluation_no_alertable.ipynb"
  },
];

const tests = [
  {
    title: "Alertable",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/05.1_inference_test_alertable.ipynb"
  },
  {
    title: "Multiclase",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/05.2_inference_test_multiclase.ipynb"
  },
  {
    title: "Híbrida",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/05.3_inference_test_hibrida.ipynb"
  },
  {
    title: "No alertable",
    link: "https://github.com/triopeligro/proyecto4geeks/blob/main/notebooks/05.4_inference_test_no_alertable.ipynb"
  },
];

export default function Preprocessing() {
  return (
    <div className="space-y-6">

      <div className="bg-white rounded-xl shadow p-5">
        <h3 className="text-base font-semibold text-blue-700 mb-2">Preprocesado y modelado</h3>
        <ul className="list-disc list-inside mt-3 space-y-2">
          {preprocessingAndModeling.map((eda) => (
            <li key={eda.title}>
              <a href={eda.link} className="text-blue-700 hover:underline" target="_blank" rel="noopener noreferrer">
                {eda.title}
              </a>
            </li>
          ))}
        </ul>
      </div>
      <div className="bg-white rounded-xl shadow p-5">
        <h3 className="text-base font-semibold text-blue-700 mb-2">Evaluaciones</h3>
        <ul className="list-disc list-inside mt-3 space-y-2">
          {evaluations.map((eda) => (
            <li key={eda.title}>
              <a href={eda.link} className="text-blue-700 hover:underline" target="_blank" rel="noopener noreferrer">
                {eda.title}
              </a>
            </li>
          ))}
        </ul>
      </div>
      <div className="bg-white rounded-xl shadow p-5">
        <h3 className="text-base font-semibold text-blue-700 mb-2">Tests</h3>
        <ul className="list-disc list-inside mt-3 space-y-2">
          {tests.map((eda) => (
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
