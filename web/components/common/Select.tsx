import { forwardRef, type SelectHTMLAttributes } from "react";

type SelectProps = SelectHTMLAttributes<HTMLSelectElement>;

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = "", children, ...props }, ref) => (
    <select
      ref={ref}
      className={`block w-full appearance-none rounded-xl border border-[rgba(148,163,184,0.25)] bg-[rgba(15,23,42,0.62)] px-4 py-2.5 text-sm text-[var(--text-strong)] shadow-[inset_0_1px_0_rgba(148,163,184,0.12)] outline-none transition focus:border-[rgba(56,189,248,0.65)] focus:ring-2 focus:ring-[rgba(56,189,248,0.25)] disabled:cursor-not-allowed disabled:opacity-60 ${className}`.trim()}
      {...props}
    >
      {children}
    </select>
  )
);

Select.displayName = "Select";

