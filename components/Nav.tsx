"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const PAGES = [
  { href: "/", label: "Dashboard" },
  { href: "/simulation", label: "Simulation" },
  { href: "/rl-lab", label: "RL Lab" },
  { href: "/analytics", label: "Portfolio Analytics" },
  { href: "/experiments", label: "Experiments" },
  { href: "/learn", label: "Learn" },
];

export function Nav() {
  const path = usePathname();
  return (
    <header className="topbar">
      <div className="brand">Multi-Agent RL: Financial Allocation</div>
      <nav aria-label="Pages">
        {PAGES.map((p) => (
          <Link key={p.href} href={p.href} className={path === p.href ? "navlink active" : "navlink"} aria-current={path === p.href ? "page" : undefined}>
            {p.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
