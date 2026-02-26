export function AnalysisSection({
  title,
  content,
}: {
  title: string;
  content: string;
}) {
  return (
    <div className="signal-card p-6">
      <h3 className="text-lg font-semibold mb-3">{title}</h3>
      <div className="text-sm text-[var(--color-text-dim)] leading-relaxed whitespace-pre-line">
        {content}
      </div>
    </div>
  );
}
