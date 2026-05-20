'use client';

type Section = 'dashboard' | 'wiki' | 'soundlist' | 'testsound';

export default function Menu({
  sidebarOpen,
  onToggleSidebar,
  section,
  onChangeSection,
}: {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  section: Section;
  onChangeSection: (section: Section) => void;
}) {
  const itemClass = (active: boolean) =>
    `block w-full rounded-xl px-3 py-2 text-left text-sm ${active ? 'bg-white/10 text-white' : 'text-slate-300 hover:bg-white/5'}`;

  return (
    <aside className={`border-r border-white/10 bg-[#0f1526] p-3 ${sidebarOpen ? 'block' : 'hidden xl:block'} xl:min-h-screen`}>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold">Menu</div>
          <div className="text-xs text-slate-400">Navegación lateral</div>
        </div>
        <button className="rounded-xl border border-white/10 px-3 py-2 text-xs hover:bg-white/5" onClick={onToggleSidebar}>
          {sidebarOpen ? 'Cerrar' : 'Abrir'}
        </button>
      </div>

      <nav className="space-y-3">
        <div>
          <div className="mb-2 text-xs uppercase tracking-[.2em] text-slate-500">Wiki</div>
          <button className={itemClass(section === 'wiki')} onClick={() => onChangeSection('wiki')}>Inicio</button>
          <button className={itemClass(false) + ' mt-1'}>Arquitectura</button>
        </div>

        <div>
          <div className="mb-2 text-xs uppercase tracking-[.2em] text-slate-500">Dashboard</div>
          <button className={itemClass(section === 'dashboard')} onClick={() => onChangeSection('dashboard')}>Inicio</button>
          <button className={itemClass(false) + ' mt-1'}>EDA</button>
        </div>

        <div>
          <div className="mb-2 text-xs uppercase tracking-[.2em] text-slate-500">SoundList</div>
          <button className={itemClass(section === 'soundlist')} onClick={() => onChangeSection('soundlist')}>Catálogo</button>
          <button className={itemClass(false) + ' mt-1'}>Etiquetas</button>
        </div>

        <div>
          <div className="mb-2 text-xs uppercase tracking-[.2em] text-slate-500">TestSound</div>
          <button className={itemClass(section === 'testsound')} onClick={() => onChangeSection('testsound')}>Playground</button>
        </div>
      </nav>
    </aside>
  );
}
