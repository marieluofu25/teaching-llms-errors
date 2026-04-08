import { Link } from 'react-router-dom'

const STAGES = [
  { id: 'residual', label: 'Residual', color: '#3B82F6', path: '/stage/residual', icon: 'analytics' },
  { id: 'export', label: 'Export', color: '#6366F1', path: '/stage/export', icon: 'upload_file' },
  { id: 'sae', label: 'SAE Encode', color: '#8B5CF6', path: '/stage/sae', icon: 'memory' },
  { id: 'diff', label: 'Diff', color: '#EC4899', path: '/stage/diff', icon: 'difference' },
  { id: 'eval', label: 'Eval', color: '#10B981', path: '/stage/eval', icon: 'task_alt' },
  { id: 'report', label: 'Report', color: '#0058BE', path: '/stage/report', icon: 'description' },
] as const

export function PipelineStepper({ activeId }: { activeId?: string }) {
  return (
    <section className="mb-12 bg-surface-container-low p-8 rounded-2xl">
      <div className="flex justify-between items-center mb-8 flex-wrap gap-4">
        <h3 className="font-headline text-xl font-bold">Pipeline integrity</h3>
        <div className="flex gap-4 text-xs font-semibold text-on-surface-variant">
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-[#10B981]" />
            Complete
          </span>
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-outline-variant/50" />
            Explore stage
          </span>
        </div>
      </div>
      <div className="relative flex justify-between items-start gap-2 md:gap-4 overflow-x-auto pb-2">
        <div className="absolute top-6 left-0 w-full h-0.5 bg-outline-variant/30 z-0 hidden md:block" />
        {STAGES.map((s) => {
          const active = s.id === activeId
          return (
            <Link
              key={s.id}
              to={s.path}
              className="relative z-10 flex flex-col items-center group min-w-[4.5rem] shrink-0"
            >
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center mb-3 ring-4 ring-surface-container-low transition-transform group-hover:scale-110 ${
                  active
                    ? 'text-white shadow-md'
                    : 'glass-panel border border-outline-variant/30 text-on-surface-variant'
                }`}
                style={
                  active
                    ? { backgroundColor: s.color, color: '#fff' }
                    : { borderColor: `${s.color}40` }
                }
              >
                <span className="material-symbols-outlined">{s.icon}</span>
              </div>
              <span
                className={`font-headline text-xs md:text-sm font-bold text-center ${
                  active ? 'text-on-surface' : 'text-on-surface-variant/80'
                }`}
              >
                {s.label}
              </span>
              <span className="text-[10px] font-bold text-[#10B981] mt-1 uppercase">Complete</span>
            </Link>
          )
        })}
      </div>
    </section>
  )
}
