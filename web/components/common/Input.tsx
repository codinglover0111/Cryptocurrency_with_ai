import { forwardRef, type InputHTMLAttributes } from "react";

type InputProps = InputHTMLAttributes<HTMLInputElement>;

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className = "", ...props }, ref) => (
    <input
      ref={ref}
      className={`block w-full rounded-xl border border-[rgba(148,163,184,0.25)] bg-[rgba(15,23,42,0.62)] px-4 py-2.5 text-sm text-[var(--text-strong)] placeholder:text-[var(--text-muted)] outline-none transition focus:border-[rgba(56,189,248,0.65)] focus:ring-2 focus:ring-[rgba(56,189,248,0.25)] disabled:cursor-not-allowed disabled:opacity-60 ${className}`.trim()}
      {...props}
    />
  )
);

Input.displayName = "Input";

