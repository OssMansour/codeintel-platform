"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { fetchModuleDoc, type WikiModuleDoc } from "@/lib/api";
import MarkdownRenderer from "@/components/MarkdownRenderer";

/**
 * Module documentation page — /wiki/[projectId]/module/[moduleName]
 */
export default function ModulePage() {
  const params = useParams();
  const projectId = params.projectId as string;
  const moduleName = params.moduleName as string;

  const [doc, setDoc] = useState<WikiModuleDoc | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!projectId || !moduleName) return;
    setLoading(true);
    setError("");
    fetchModuleDoc(projectId, moduleName)
      .then(setDoc)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [projectId, moduleName]);

  if (loading) {
    return <p className="text-slate-400 mt-10">Loading module…</p>;
  }

  if (error) {
    return (
      <div className="mt-10">
        <h1 className="text-2xl font-bold mb-4">{moduleName}</h1>
        <p className="text-red-400 mb-4">{error}</p>
        <Link
          href={`/wiki/${encodeURIComponent(projectId)}`}
          className="text-blue-400 hover:underline"
        >
          ← Back to overview
        </Link>
      </div>
    );
  }

  return (
    <div>
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-slate-500 mb-6">
        <Link
          href={`/wiki/${encodeURIComponent(projectId)}`}
          className="hover:text-blue-400 transition-colors"
        >
          {projectId}
        </Link>
        <span>/</span>
        <span className="text-slate-300">{moduleName}</span>
      </nav>

      <h1 className="text-3xl font-bold mb-6">{moduleName}</h1>

      {doc && <MarkdownRenderer content={doc.content} />}
    </div>
  );
}
