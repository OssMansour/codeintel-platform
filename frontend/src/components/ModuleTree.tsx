"use client";

import Link from "next/link";
import type { WikiModuleNode } from "@/lib/api";

/**
 * Recursive tree sidebar for navigating the module hierarchy.
 */
export default function ModuleTree({
  modules,
  projectId,
  activeModule,
}: {
  modules: Record<string, WikiModuleNode>;
  projectId: string;
  activeModule?: string;
}) {
  return (
    <nav className="text-sm">
      <ul className="space-y-0.5">
        {Object.entries(modules).map(([name, node]) => (
          <TreeNode
            key={name}
            name={name}
            node={node}
            projectId={projectId}
            activeModule={activeModule}
            depth={0}
          />
        ))}
      </ul>
    </nav>
  );
}

function TreeNode({
  name,
  node,
  projectId,
  activeModule,
  depth,
}: {
  name: string;
  node: WikiModuleNode;
  projectId: string;
  activeModule?: string;
  depth: number;
}) {
  const isActive = activeModule === name;
  const hasChildren = node.children && Object.keys(node.children).length > 0;
  const pl = depth * 12;

  return (
    <li>
      <Link
        href={`/wiki/${encodeURIComponent(projectId)}/module/${encodeURIComponent(name)}`}
        className={`
          flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors
          ${isActive
            ? "bg-blue-600/20 text-blue-300 font-medium"
            : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
          }
        `}
        style={{ paddingLeft: `${12 + pl}px` }}
      >
        <span className="text-xs opacity-60">
          {hasChildren ? "📁" : "📄"}
        </span>
        <span className="truncate">{name}</span>
        {!node.has_doc && (
          <span className="ml-auto text-[10px] text-slate-600">(no doc)</span>
        )}
      </Link>

      {hasChildren && (
        <ul className="space-y-0.5">
          {Object.entries(node.children).map(([childName, childNode]) => (
            <TreeNode
              key={childName}
              name={childName}
              node={childNode}
              projectId={projectId}
              activeModule={activeModule}
              depth={depth + 1}
            />
          ))}
        </ul>
      )}
    </li>
  );
}
