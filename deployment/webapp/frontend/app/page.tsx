'use client';

import { useMemo, useState } from 'react';
import Menu from '@/components/layout/Menu';
import DashboardHome from '@/components/dashboard/DashboardHome';
import WikiHome from '@/components/wiki/WikiHome';
import SoundListHome from '@/components/soundlist/SoundListHome';
import TestSoundHome from '@/components/testsound/TestSoundHome';
import MobileView from '@/components/mobile/MobileView';

type Section = 'dashboard' | 'wiki' | 'soundlist' | 'testsound';

export default function Page() {
  const [section, setSection] = useState<Section>('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [phoneOpen, setPhoneOpen] = useState(true);

  const content = useMemo(() => {
    switch (section) {
      case 'wiki':
        return <WikiHome />;
      case 'soundlist':
        return <SoundListHome />;
      case 'testsound':
        return <TestSoundHome />;
      default:
        return <DashboardHome />;
    }
  }, [section]);

  return (
    <div className={`min-h-screen ${phoneOpen ? 'xl:grid xl:grid-cols-[260px_minmax(0,1fr)_380px]' : 'xl:grid xl:grid-cols-[260px_minmax(0,1fr)]'} bg-[#0b1020]`}>
      <Menu
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(v => !v)}
        section={section}
        onChangeSection={setSection}
      />

      <main className={`min-h-screen p-4 md:p-6 ${sidebarOpen ? '' : 'xl:col-span-2'}`}>
        <div className="mb-4 flex items-center justify-between gap-3 rounded-2xl border border-white/10 bg-[#11182a] p-4">
          <div>
            <h1 className="text-2xl font-semibold">DataScience Multifunction</h1>
            <p className="text-sm text-slate-400">EDA, modelo, wiki, sonidos y vista móvil en un solo panel.</p>
          </div>
          <button
            className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm hover:bg-white/10"
            onClick={() => setPhoneOpen(v => !v)}
          >
            {phoneOpen ? 'Ocultar móvil' : 'Mostrar móvil'}
          </button>
        </div>

        <div className="grid gap-4">
          {content}
        </div>
      </main>

      {phoneOpen && (
        <aside className="border-l border-white/10 bg-[#0f1526] p-3 xl:min-h-screen">
          <MobileView />
        </aside>
      )}
    </div>
  );
}
