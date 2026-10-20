"use client";

import { useEffect, type ReactNode } from "react";

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  useEffect(() => {
    document.title = `${title} | Allocation Lab`;
  }, [title]);
  return (
    <div className="page-header">
      <h1>{title}</h1>
      {children ? <p className="muted">{children}</p> : null}
    </div>
  );
}
