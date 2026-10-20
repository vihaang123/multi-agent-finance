"use client";

export default function PageError({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div className="fatal">
      <h1>This page hit a problem</h1>
      <p className="muted">
        Something went wrong while drawing the page ({error.message || "unknown error"}). Your saved agents are untouched.
      </p>
      <div className="toolbar">
        <button type="button" className="btn primary" onClick={reset}>Try again</button>
        <a className="btn" href="/">Back to the dashboard</a>
      </div>
    </div>
  );
}
