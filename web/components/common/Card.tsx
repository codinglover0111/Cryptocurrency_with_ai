import type { PropsWithChildren } from "react";

interface CardProps extends PropsWithChildren {
  className?: string;
}

export function Card({ className = "", children }: CardProps) {
  return (
    <div className="relative overflow-hidden rounded-[18px] border border-[rgba(148,163,184,0.14)] bg-[rgba(17,29,52,0.92)] p-6 shadow-[0_18px_36px_rgba(2,6,23,0.35)] backdrop-blur-xl">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(120deg,rgba(56,189,248,0.08),transparent_58%)] opacity-70" />
      <div className={`relative z-10 ${className}`.trim()}>{children}</div>
    </div>
  );
}
