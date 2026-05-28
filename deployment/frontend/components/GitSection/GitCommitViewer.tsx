'use client';

import { useEffect, useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { es } from 'date-fns/locale';
import styles from "./GitCommitViewer.module.css";
import { getGithubCommits, Commit } from "@/services/api";

export function GitCommitViewer() {
  const [commits, setCommits] = useState<Commit[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCommit, setSelectedCommit] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCommits = async () => {
      try {
        setLoading(true);

        const commits = await getGithubCommits(12);

        setCommits(commits);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Error desconocido");
        console.error("Error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchCommits();
  }, []);

  if (error) {
    return (
      <div className={`${styles["error-box"]} flex items-center justify-center h-screen box black-box`}>
        <div className="text-center p-6">
          <div className="text-5xl mb-4">⚠️</div>
          <p className="text-gray-400 font-medium mb-2">Error al cargar commits</p>
          <p className="text-gray-400 text-sm">{error}</p>
          <p className="text-gray-500 text-xs mt-4">
            Verifica que el backend esté corriendo en localhost:8000
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`${styles["commits__container"]} flex items-center justify-center min-h-screen  from-slate-900 via-slate-800 to-slate-900 p-4`}>
      {/* Contenedor principal 9:16 */}
      <div className="w-full max-w-sm aspect-[9/16] box black-box overflow-hidden flex flex-col">

        {/* Header */}
        <div className=" from-blue-600 to-cyan-600 px-6 py-8 text-white">
          <h1 className="text-2xl font-black tracking-tight">Proyecto4geeks-Mivia</h1>
          <p className="text-blue-100 text-sm mt-2 font-medium">Últimos cambios</p>
          <div className="mt-4 flex gap-2 text-xs">
            <span className="bg-white/20 px-3 py-1 rounded-full backdrop-blur">main</span>
            <span className="bg-white/20 px-3 py-1 rounded-full backdrop-blur">
              {commits.length} commits
            </span>
          </div>
        </div>

        {/* Commits List */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
          {loading ? (
            <div className="flex items-center justify-center box blue-box h-full">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-slate-600 text-sm font-medium">Cargando commits...</p>
              </div>
            </div>
          ) : commits.length === 0 ? (
            <div className="flex items-center justify-center blue-box h-full text-center">
              <p className="text-slate-400">No hay commits disponibles</p>
            </div>
          ) : (
            commits.map((commit, idx) => (
              <div
                key={commit.hash_full}
                onClick={() => setSelectedCommit(
                  selectedCommit === commit.hash_full ? null : commit.hash_full
                )}
                className={`${styles.commit} blue-box
                  relative p-4 rounded-xl cursor-pointer transition-all duration-300
                  ${selectedCommit === commit.hash_full
                    ? 'bg-blue-50'
                    : 'bg-slate-50'
                  }
                `}
                style={{
                  animation: `slideIn 0.5s ease-out ${idx * 50}ms both`
                }}
              >
                {/* Indicador de rama */}
                <div className="absolute top-2 right-2 w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>

                {/* Hash y fecha */}
                <div className="flex items-center justify-between mb-2">
                  <code className="text-xs font-mono bg-slate-900 text-cyan-400 px-2 py-1 rounded">
                    {commit.hash}
                  </code>
                  <span className="text-xs text-slate-500 font-medium">
                    {formatDistanceToNow(new Date(commit.date), {
                      addSuffix: true,
                      locale: es
                    })}
                  </span>
                </div>

                {/* Mensaje */}
                <p className="text-sm font-semibold text-slate-900 mb-2 line-clamp-2">
                  {commit.message}
                </p>

                {/* Autor */}
                <p className="text-xs text-slate-600 font-medium mb-3">
                  by {commit.author}
                </p>

                {/* Estadísticas (expandido) */}
                {selectedCommit === commit.hash_full && (
                  <div className="mt-3 pt-3 border-t border-slate-200 space-y-2 animate-in fade-in duration-300">
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="bg-emerald-50 rounded py-1.5 px-1">
                        <p className="text-xs font-bold text-emerald-600">+{commit.additions}</p>
                        <p className="text-xs text-emerald-500 font-medium">agregadas</p>
                      </div>
                      <div className="bg-red-50 rounded py-1.5 px-1">
                        <p className="text-xs font-bold text-red-600">{commit.deletions}</p>
                        <p className="text-xs text-red-500 font-medium">eliminadas</p>
                      </div>
                      <div className="bg-blue-50 rounded py-1.5 px-1">
                        <p className="text-xs font-bold text-blue-600">{commit.files_changed}</p>
                        <p className="text-xs text-blue-500 font-medium">archivos</p>
                      </div>
                    </div>
                    <a
                      href={commit.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-center text-xs font-semibold text-blue-600 hover:text-blue-700 mt-2 py-1 hover:bg-blue-50 rounded transition-colors"
                    >
                      Ver en GitHub →
                    </a>
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className=" px-4 py-4 text-center">
          <p className="text-xs text-slate-500 font-medium">
            📊 Actualizado en tiempo real
          </p>
          <p className="text-xs text-slate-400 mt-1">
            Haz click en un commit para más detalles
          </p>
        </div>
      </div>

      <style jsx>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
