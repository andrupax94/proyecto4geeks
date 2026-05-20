export default function MobileView() {
  return (
    <div className="sticky top-3 overflow-hidden rounded-[2rem] border border-white/10 bg-[#0b1020] p-3 shadow-2xl">
      <div className="mx-auto w-[290px] rounded-[2rem] border border-white/10 bg-[#11182a] p-3">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold">Mobile view</div>
            <div className="text-xs text-slate-400">Simulación de app</div>
          </div>
          <div className="rounded-full bg-white/10 px-2 py-1 text-[10px] text-slate-300">LIVE</div>
        </div>

        <div className="space-y-3">
          <Tile title="Predict" subtitle="Audio / clase" />
          <Tile title="Alert" subtitle="Trigger / strike" />
          <Tile title="Status" subtitle="Online" />
        </div>
      </div>
    </div>
  );
}

function Tile({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
      <div className="text-sm font-medium">{title}</div>
      <div className="mt-1 text-xs text-slate-400">{subtitle}</div>
    </div>
  );
}
