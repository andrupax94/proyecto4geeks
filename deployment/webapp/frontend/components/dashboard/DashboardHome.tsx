export default function DashboardHome() {
  return (
    <section className="grid gap-4">
      <div className="rounded-2xl border border-white/10 bg-[#11182a] p-5">
        <h2 className="text-xl font-semibold">Dashboard EDA</h2>
        <p className="mt-2 text-sm text-slate-400">Resumen de la build de análisis exploratorio.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <Card title="Resumen" value="12,480 audios" />
        <Card title="Calidad" value="8.7 / 10" />
        <Card title="Clases" value="24 etiquetas" />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Gráfico 1" />
        <Panel title="Gráfico 2" />
      </div>

      <Panel title="Notas del EDA" tall />
    </section>
  );
}

function Card({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[#11182a] p-5">
      <div className="text-sm text-slate-400">{title}</div>
      <div className="mt-2 text-2xl font-semibold">{value}</div>
    </div>
  );
}

function Panel({ title, tall = false }: { title: string; tall?: boolean }) {
  return (
    <div className={`rounded-2xl border border-white/10 bg-[#11182a] p-5 ${tall ? 'min-h-[280px]' : 'min-h-[220px]'}`}>
      <div className="text-sm text-slate-400">{title}</div>
      <div className="mt-4 h-full rounded-xl border border-dashed border-white/10 bg-white/5" />
    </div>
  );
}
