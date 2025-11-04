"use client";

import React, { type ButtonHTMLAttributes, forwardRef } from "react";

type ButtonVariant = "primary" | "secondary" | "ghost";
type ButtonSize = "sm" | "md";

const VARIANT_STYLES: Record<ButtonVariant, string> = {
  primary:
    "border-[rgba(56,189,248,0.45)] bg-[rgba(14,165,233,0.92)] text-slate-950 hover:bg-[rgba(14,165,233,1)]",
  secondary:
    "border-[rgba(148,163,184,0.35)] bg-[rgba(15,23,42,0.65)] text-[var(--text-strong)] hover:border-[rgba(148,163,184,0.5)]",
  ghost:
    "border-transparent bg-[rgba(15,23,42,0.45)] text-[var(--text-muted)] hover:bg-[rgba(15,23,42,0.65)]",
};

const SIZE_STYLES: Record<ButtonSize, string> = {
  sm: "h-9 rounded-full px-4 text-xs",
  md: "h-10 rounded-full px-5 text-sm",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  asChild?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className = "",
      variant = "primary",
      size = "md",
      asChild = false,
      children,
      ...props
    },
    ref
  ) => {
    const base =
      "inline-flex items-center justify-center gap-2 border font-semibold transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-400 disabled:cursor-not-allowed disabled:opacity-60 shadow-[0_14px_28px_rgba(8,47,73,0.35)]";
    const variantClass = VARIANT_STYLES[variant] ?? VARIANT_STYLES.primary;
    const sizeClass = SIZE_STYLES[size] ?? SIZE_STYLES.md;
    const combinedClassName =
      `${base} ${variantClass} ${sizeClass} ${className}`.trim();

    if (asChild && React.isValidElement(children)) {
      return React.cloneElement(children, {
        className: `${combinedClassName} ${
          children.props?.className ?? ""
        }`.trim(),
        ...props,
      });
    }

    return (
      <button ref={ref} className={combinedClassName} {...props}>
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";
