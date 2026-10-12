import { Fragment, type ReactNode } from "react";

// A tiny, safe Markdown subset (no raw HTML): **bold**, *italic*, `code`, "- " bullet lists, blank-line paragraphs.
const TOKEN = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;

export function Inline({ text }: { text: string }): ReactNode {
  const parts = text.split(TOKEN);
  return (
    <>
      {parts.map((p, i) => {
        if (!p) return null;
        if (p.startsWith("`") && p.endsWith("`") && p.length > 1) return <code key={i}>{p.slice(1, -1)}</code>;
        if (p.startsWith("**") && p.endsWith("**") && p.length > 4) return <strong key={i}>{p.slice(2, -2)}</strong>;
        if (p.startsWith("*") && p.endsWith("*") && p.length > 2) return <em key={i}>{p.slice(1, -1)}</em>;
        return <Fragment key={i}>{p}</Fragment>;
      })}
    </>
  );
}

function Lines({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <>
      {lines.map((l, i) => (
        <Fragment key={i}>
          {i > 0 && <br />}
          <Inline text={l} />
        </Fragment>
      ))}
    </>
  );
}

export function Markdown({ text, className = "" }: { text: string; className?: string }) {
  const paragraphs = text.trim().split(/\n\s*\n/);
  return (
    <div className={`md ${className}`}>
      {paragraphs.map((para, i) => {
        const lines = para.split("\n");
        if (lines.every((l) => /^\s*-\s+/.test(l))) {
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>
                  <Inline text={l.replace(/^\s*-\s+/, "")} />
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i}>
            <Lines text={para} />
          </p>
        );
      })}
    </div>
  );
}
