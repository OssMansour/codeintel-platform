"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { fetchModuleTree, type WikiModuleTree } from "@/lib/api";
import ModuleTree from "@/components/ModuleTree";
import SearchBar from "@/components/SearchBar";
import ChatSidebar from "@/components/ChatSidebar";

/**
 * Wiki project layout — sidebar + content area.
 * All /wiki/[projectId]/* pages are children of this layout.
 */
export default function WikiLayout({ children }: { children: React.ReactNode }) {
  const params = useParams();
  const projectId = params.projectId as string;
  const moduleName = params.moduleName as string | undefined;

  const [tree, setTree] = useState<WikiModuleTree | null>(null);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    if (!projectId) return;
    fetchModuleTree(projectId)
      .then(setTree)
      .catch((e) => setError(e.message));
  }, [projectId]);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`
          ${sidebarOpen ? "w-72" : "w-0"} flex-shrink-0
          bg-slate-900 border-r border-slate-700 transition-all duration-200
          flex flex-col overflow-hidden
        `}
      >
        {/* Sidebar header */}
        <div className="p-4 border-b border-slate-700">
          <Link
            href={`/wiki/${encodeURIComponent(projectId)}`}
            className="text-lg font-bold text-white hover:text-blue-300 transition-colors"
          >
            📘 {projectId}
          </Link>
          <p className="text-xs text-slate-500 mt-1">
            {tree?.generated_at
              ? `Generated ${new Date(tree.generated_at).toLocaleDateString()}`
              : "Wiki"}
          </p>
        </div>

        {/* Search */}
        <div className="p-3">
          <SearchBar projectId={projectId} />
        </div>

        {/* Nav links */}
        <div className="px-3 mb-2">
          <Link
            href={`/wiki/${encodeURIComponent(projectId)}`}
            className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm
                       text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            🏠 Overview
          </Link>
        </div>

        {/* Module tree */}
        <div className="flex-1 overflow-y-auto px-3 pb-4">
          {error && <p className="text-red-400 text-xs p-2">{error}</p>}
          {tree && (
            <ModuleTree
              modules={tree.modules}
              projectId={projectId}
              activeModule={moduleName}
            />
          )}
        </div>
      </aside>

      {/* Sidebar toggle */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="absolute top-3 left-2 z-20 text-slate-500 hover:text-white
                   bg-slate-800 rounded px-1.5 py-0.5 text-xs border border-slate-700"
        style={{ left: sidebarOpen ? "290px" : "4px" }}
      >
        {sidebarOpen ? "◀" : "▶"}
      </button>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-8 py-8">{children}</div>
      </main>

      {/* Floating chat */}
      <ChatSidebar projectId={projectId} />
    </div>
  );
}
