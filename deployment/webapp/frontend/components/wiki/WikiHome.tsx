export default function WikiHome() {
  return (
    <section className="grid gap-4">
      <div className="rounded-2xl border border-white/10 bg-[#11182a] p-5">
        <h2 className="text-xl font-semibold">Wiki interna</h2>
        <p className="mt-2 text-sm text-slate-400">Documentación corta del proyecto, arquitectura y uso.</p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <DocCard title="Inicio" />
        <DocCard title="Arquitectura" />
      </div>
    </section>
  );
}

function DocCard({ title }: { title: string }) {
  return (
    <article className="rounded-2xl border border-white/10 bg-[#11182a] p-5">
      <div className="text-base font-semibold">{title}</div>
      <p className="mt-2 text-sm text-slate-400">
        Bloque de documentación para explicar decisiones, pipeline, rutas y despliegue.
      </p>
    </article>
  );
}
