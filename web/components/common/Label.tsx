import type { LabelHTMLAttributes, PropsWithChildren } from "react";

type LabelProps = PropsWithChildren<LabelHTMLAttributes<HTMLLabelElement>>;

export function Label({ className = "", children, ...props }: LabelProps) {
  return (
    <label
      className={`flex flex-col gap-2 text-[13px] font-semibold uppercase tracking-[0.12em] text-[var(--text-muted)] ${className}`.trim()}
      {...props}
    >
      {children}
    </label>
  );
}

