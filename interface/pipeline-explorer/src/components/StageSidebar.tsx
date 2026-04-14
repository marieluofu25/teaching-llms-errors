import { NavLink } from 'react-router-dom'
import { MaterialIcon } from './MaterialIcon'

const LINKS = [
  { to: '/poster', label: 'Poster', icon: 'newspaper', color: '#0058BE' },
  { to: '/stage/residual', label: 'Residual', icon: 'analytics', color: '#3B82F6' },
  { to: '/stage/export', label: 'Export', icon: 'upload_file', color: '#6366F1' },
  { to: '/stage/sae', label: 'SAE Encode', icon: 'memory', color: '#8B5CF6' },
  { to: '/stage/diff', label: 'Diff', icon: 'difference', color: '#EC4899' },
  { to: '/stage/eval', label: 'Eval', icon: 'task_alt', color: '#10B981' },
  { to: '/stage/report', label: 'Report', icon: 'description', color: '#0058BE' },
] as const

export function StageSidebar() {
  return (
    <aside className="fixed left-0 top-0 h-full flex flex-col z-40 bg-surface-container-low w-64 border-none">
      <div className="px-8 py-8">
        <div className="text-lg font-extrabold text-on-surface font-headline mb-1">Pipeline stages</div>
        <div className="text-[11px] uppercase tracking-widest text-on-surface/50 font-bold">
          LLM error analysis
        </div>
      </div>
      <nav className="flex-1 px-4 space-y-1">
        {LINKS.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            className={({ isActive }) =>
              [
                'flex items-center px-4 py-3 rounded-l-lg transition-all duration-200 group',
                isActive
                  ? 'bg-surface-container-lowest text-primary shadow-sm border-l-4 font-semibold'
                  : 'text-on-surface/70 hover:bg-surface hover:translate-x-0.5',
              ].join(' ')
            }
            style={({ isActive }) =>
              isActive ? { borderLeftColor: l.color, borderLeftWidth: 4 } : undefined
            }
          >
            <MaterialIcon name={l.icon} className="mr-3" />
            <span className="text-sm">{l.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="p-6">
        <NavLink
          to="/"
          className="w-full py-3 bg-primary text-white rounded-xl font-bold text-sm flex items-center justify-center gap-2 hover:bg-primary-container transition-colors"
        >
          <MaterialIcon name="arrow_back" className="text-sm" />
          Run overview
        </NavLink>
      </div>
    </aside>
  )
}
