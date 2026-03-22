"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchProjects } from "@/lib/api";

export default function HomePage() {
  const [projects, setProjects] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchProjects()
      .then(setProjects)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold mb-2">
        📘 {process.env.NEXT_PUBLIC_APP_NAME || "CodeIntel"} Wiki
      </h1>
      <p className="text-slate-400 mb-10">
        AI-generated code documentation browser
      </p>

      {loading && <p className="text-slate-400">Loading projects…</p>}
      {error && <p className="text-red-400">{error}</p>}

      {!loading && projects.length === 0 && !error && (
        <p className="text-slate-500">
          No wikis generated yet. Trigger a wiki generation via the API.
        </p>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 w-full max-w-4xl">
        {projects.map((pid) => (
          <Link
            key={pid}
            href={`/wiki/${encodeURIComponent(pid)}`}
            className="block bg-slate-800 hover:bg-slate-700 border border-slate-700
                       rounded-xl p-6 transition-colors"
          >
            <h2 className="text-lg font-semibold mb-1">{pid}</h2>
            <p className="text-sm text-slate-400">View wiki →</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
