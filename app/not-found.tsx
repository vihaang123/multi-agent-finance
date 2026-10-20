import Link from "next/link";

export default function NotFound() {
  return (
    <div className="fatal">
      <h1>That page does not exist</h1>
      <p className="muted">Use the sidebar to open the dashboard, simulation, RL Lab, analytics, experiments or learn pages.</p>
      <Link className="btn primary" href="/">Back to the dashboard</Link>
    </div>
  );
}
