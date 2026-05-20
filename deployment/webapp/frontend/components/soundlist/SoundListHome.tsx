export default function SoundListHome() {
  const sounds = ['gun_shot_01.wav', 'car_crash_02.wav', 'alarm_07.wav'];

  return (
    <section className="grid gap-4">
      <div className="rounded-2xl border border-white/10 bg-[#11182a] p-5">
        <h2 className="text-xl font-semibold">SoundList</h2>
        <p className="mt-2 text-sm text-slate-400">Listado base de sonidos, etiquetas y estado.</p>
      </div>

      <div className="rounded-2xl border border-white/10 bg-[#11182a] p-5">
        <ul className="space-y-3 text-sm">
          {sounds.map(sound => (
            <li key={sound} className="flex items-center justify-between rounded-xl bg-white/5 px-3 py-2">
              <span>{sound}</span>
              <span className="text-slate-400">ready</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
