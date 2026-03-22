"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

/**
 * Renders Markdown with Mermaid diagram support.
 * Mermaid fenced blocks are rendered client-side via the mermaid library.
 */
export default function MarkdownRenderer({ content }: { content: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initialise Mermaid after mount
    import("mermaid").then((m) => {
      m.default.initialize({
        startOnLoad: false,
        theme: "dark",
        themeVariables: {
          darkMode: true,
          background: "#1e293b",
          primaryColor: "#3b82f6",
          primaryTextColor: "#f1f5f9",
          lineColor: "#64748b",
        },
      });
      // Render all .mermaid blocks in the container
      if (containerRef.current) {
        const blocks = containerRef.current.querySelectorAll(".mermaid-source");
        blocks.forEach(async (el, i) => {
          try {
            const { svg } = await m.default.render(`mermaid-${i}`, el.textContent || "");
            el.innerHTML = svg;
            el.classList.remove("mermaid-source");
            el.classList.add("mermaid");
          } catch {
            el.classList.add("text-red-400", "text-xs");
            el.textContent = `[Mermaid render error] ${el.textContent?.slice(0, 80)}`;
          }
        });
      }
    });
  }, [content]);

  const components: Components = {
    code({ className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || "");
      const lang = match?.[1];

      if (lang === "mermaid") {
        return (
          <div className="mermaid-source bg-slate-800 rounded-lg p-4 mb-4 text-center">
            {String(children).replace(/\n$/, "")}
          </div>
        );
      }

      // Inline code
      if (!className) {
        return (
          <code className="bg-slate-700 px-1.5 py-0.5 rounded text-sm text-blue-300" {...props}>
            {children}
          </code>
        );
      }

      return (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
  };

  return (
    <div ref={containerRef} className="prose-wiki max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
