"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { fetchOverview, type WikiOverview } from "@/lib/api";
import MarkdownRenderer from "@/components/MarkdownRenderer";

/**
 * Project overview page — /wiki/[projectId]
 */
export default function WikiOverviewPage() {
  const params = useParams();
  const projectId = params.projectId as string;

  const [overview, setOverview] = useState<WikiOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!projectId) return;
    setLoading(true);
    fetchOverview(projectId)
      .then(setOverview)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [projectId]);

  if (loading) {
    return <p className="text-slate-400 mt-10">Loading overview…</p>;
  }

  if (error) {
    return (
      <div className="mt-10">
        <h1 className="text-2xl font-bold mb-4">Project: {projectId}</h1>
        <p className="text-slate-400">
          No overview generated yet for this project. Trigger wiki generation via the API.
        </p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">
        {overview?.repo_name || projectId} — Overview
      </h1>
      {overview && <MarkdownRenderer content={overview.content} />}
    </div>
  );
}
