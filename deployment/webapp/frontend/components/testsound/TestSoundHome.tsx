export default function TestSoundHome() {
  return (
    <section className="grid gap-4">
      <div className="rounded-2xl border border-white/10 bg-[#11182a] p-5">
        <h2 className="text-xl font-semibold">TestSound</h2>
        <p className="mt-2 text-sm text-slate-400">Zona de prueba para cargar un audio y obtener respuesta del backend.</p>
      </div>

      <div className="rounded-2xl border border-white/10 bg-[#11182a] p-5">
        <div className="h-40 rounded-xl border border-dashed border-white/10 bg-white/5" />
        <button className="mt-4 rounded-xl bg-white px-4 py-2 text-sm font-medium text-black">
          Cargar / Probar
        </button>
      </div>
    </section>
  );
}
