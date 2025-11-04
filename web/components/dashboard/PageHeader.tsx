interface PageHeaderProps {
  title: string;
  subtitle?: string;
  badgeText?: string;
}

export function PageHeader({ title, subtitle, badgeText }: PageHeaderProps) {
  return (
    <header className="page-header card">
      <div className="page-heading">
        <h1 className="page-title">{title}</h1>
        {subtitle ? <p className="page-subtitle">{subtitle}</p> : null}
      </div>
      {badgeText ? (
        <div className="page-meta">
          <span className="chip chip--accent">{badgeText}</span>
        </div>
      ) : null}
    </header>
  );
}
