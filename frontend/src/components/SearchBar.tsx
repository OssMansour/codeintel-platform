"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { searchWiki, type WikiSearchResult } from "@/lib/api";

/**
 * Search bar for wiki keyword search.
 */
export default function SearchBar({ projectId }: { projectId: string }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<WikiSearchResult[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [searching, setSearching] = useState(false);
  const router = useRouter();

  async function handleSearch(q: string) {
    setQuery(q);
    if (q.trim().length < 2) {
      setResults([]);
      setShowDropdown(false);
      return;
    }
    setSearching(true);
    try {
      const res = await searchWiki(projectId, q.trim());
      setResults(res.results);
      setShowDropdown(res.results.length > 0);
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  return (
    <div className="relative">
      <input
        value={query}
        onChange={(e) => handleSearch(e.target.value)}
        onFocus={() => results.length > 0 && setShowDropdown(true)}
        onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
        placeholder="Search wiki…"
        className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2
                   text-sm text-white placeholder:text-slate-500 focus:outline-none
                   focus:border-blue-500"
      />
      {searching && (
        <span className="absolute right-3 top-2.5 text-xs text-slate-500">⏳</span>
      )}

      {showDropdown && (
        <div className="absolute z-30 mt-1 w-full bg-slate-800 border border-slate-700
                        rounded-lg shadow-xl max-h-80 overflow-y-auto">
          {results.map((r, i) => (
            <button
              key={i}
              onMouseDown={() => {
                router.push(
                  `/wiki/${encodeURIComponent(projectId)}/module/${encodeURIComponent(r.module_name)}`
                );
                setShowDropdown(false);
              }}
              className="block w-full text-left px-4 py-3 hover:bg-slate-700 border-b
                         border-slate-700 last:border-0 transition-colors"
            >
              <span className="text-sm font-medium text-blue-300">
                {r.module_name}
              </span>
              <p className="text-xs text-slate-400 mt-0.5 line-clamp-2">
                {r.snippet}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
