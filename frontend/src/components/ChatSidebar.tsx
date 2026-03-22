"use client";

import { useState, useRef, useEffect } from "react";
import { agentQueryStream, type ChatSource } from "@/lib/api";
import MarkdownRenderer from "./MarkdownRenderer";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
}

/**
 * Collapsible chat sidebar for asking the agent questions.
 */
export default function ChatSidebar({ projectId }: { projectId: string }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const q = input.trim();
    if (!q || streaming) return;
    setInput("");

    const userMsg: Message = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setStreaming(true);

    const assistantMsg: Message = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      for await (const event of agentQueryStream(q, projectId)) {
        if (event.type === "answer") {
          assistantMsg.content += event.content as string;
          setMessages((prev) => [...prev.slice(0, -1), { ...assistantMsg }]);
        } else if (event.type === "sources") {
          assistantMsg.sources = event.content as ChatSource[];
          setMessages((prev) => [...prev.slice(0, -1), { ...assistantMsg }]);
        }
      }
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : "Unknown error";
      assistantMsg.content += `\n\n⚠️ ${errMsg}`;
      setMessages((prev) => [...prev.slice(0, -1), { ...assistantMsg }]);
    } finally {
      setStreaming(false);
    }
  }

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setOpen(!open)}
        className="fixed bottom-6 right-6 z-50 bg-blue-600 hover:bg-blue-500
                   text-white rounded-full w-14 h-14 flex items-center justify-center
                   shadow-lg transition-colors text-xl"
        title="Ask CodeIntel"
      >
        {open ? "✕" : "💬"}
      </button>

      {/* Panel */}
      {open && (
        <div className="fixed bottom-24 right-6 z-40 w-96 max-h-[70vh] flex flex-col
                        bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 border-b border-slate-700 bg-slate-800">
            <h3 className="font-semibold text-sm">Ask CodeIntel</h3>
            <p className="text-xs text-slate-400">
              AI code intelligence agent — searches code, docs &amp; incidents
            </p>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[200px]">
            {messages.length === 0 && (
              <p className="text-sm text-slate-500 text-center mt-8">
                Ask anything about the codebase…
              </p>
            )}
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`text-sm ${
                  msg.role === "user"
                    ? "text-right"
                    : "text-left"
                }`}
              >
                <div
                  className={`inline-block max-w-[90%] px-3 py-2 rounded-lg ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-slate-800 text-slate-200"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <MarkdownRenderer content={msg.content || "Thinking…"} />
                  ) : (
                    msg.content
                  )}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-1 text-xs text-slate-500">
                    {msg.sources.length} source{msg.sources.length !== 1 && "s"} cited
                  </div>
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center gap-2 p-3 border-t border-slate-700"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about the code…"
              disabled={streaming}
              className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-3 py-2
                         text-sm text-white placeholder:text-slate-500 focus:outline-none
                         focus:border-blue-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={streaming || !input.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40
                         text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              {streaming ? "…" : "Send"}
            </button>
          </form>
        </div>
      )}
    </>
  );
}
