export function LoadState({
  loading,
  error,
  children,
}: {
  loading: boolean
  error: Error | null
  children: React.ReactNode
}) {
  if (loading) {
    return (
      <div className="rounded-xl bg-surface-container-low p-8 text-on-surface-variant text-sm">
        Loading pipeline artifacts…
      </div>
    )
  }
  if (error) {
    return (
      <div className="rounded-xl bg-error/10 text-error p-6 text-sm border border-error/20">
        <p className="font-bold mb-1">Could not load data</p>
        <p>{error.message}</p>
        <p className="mt-3 text-on-surface-variant">
          Start the dev server from <code className="text-xs bg-surface-container-low px-1 rounded">interface/pipeline-explorer</code> so{' '}
          <code className="text-xs bg-surface-container-low px-1 rounded">/pipeline_2026</code> is mounted.
        </p>
      </div>
    )
  }
  return <>{children}</>
}
