"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        h1: ({ children }) => <h1 className="mb-2 mt-3 text-lg font-bold first:mt-0">{children}</h1>,
        h2: ({ children }) => <h2 className="mb-1.5 mt-3 text-base font-semibold first:mt-0">{children}</h2>,
        h3: ({ children }) => <h3 className="mb-1 mt-2 text-sm font-semibold first:mt-0">{children}</h3>,
        ul: ({ children }) => <ul className="mb-2 ml-4 list-disc space-y-1">{children}</ul>,
        ol: ({ children }) => <ol className="mb-2 ml-4 list-decimal space-y-1">{children}</ol>,
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        code: ({ className, children }) => {
          if (className) {
            return (
              <code className="mb-2 mt-1 block max-w-full overflow-x-auto rounded-md bg-muted p-2.5 text-xs leading-relaxed">
                {children}
              </code>
            );
          }
          return <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono">{children}</code>;
        },
        pre: ({ children }) => <pre className="mb-2 mt-1">{children}</pre>,
        blockquote: ({ children }) => (
          <blockquote className="my-2 border-l-2 border-primary/30 pl-3 text-muted-foreground">{children}</blockquote>
        ),
        table: ({ children }) => (
          <div className="my-2 overflow-x-auto">
            <table className="w-full text-xs">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border-b bg-muted/50 px-2 py-1.5 text-left font-semibold">{children}</th>
        ),
        td: ({ children }) => <td className="border-b px-2 py-1.5">{children}</td>,
        hr: () => <hr className="my-3 border-border/50" />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
