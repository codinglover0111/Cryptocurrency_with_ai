import type { PropsWithChildren } from "react";

interface BadgeProps extends PropsWithChildren {
  variant?: "default" | "accent";
  className?: string;
}

export function Badge({
  variant = "default",
  className = "",
  children,
}: BadgeProps) {
  const base =
    "inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] backdrop-blur";
  const styles =
    variant === "accent"
      ? "border-[rgba(56,189,248,0.45)] bg-[rgba(56,189,248,0.15)] text-sky-300"
      : "border-[rgba(148,163,184,0.24)] bg-[rgba(15,23,42,0.72)] text-[var(--text-muted)]";

  return (
    <span className={`${base} ${styles} ${className}`.trim()}>{children}</span>
  );
}
