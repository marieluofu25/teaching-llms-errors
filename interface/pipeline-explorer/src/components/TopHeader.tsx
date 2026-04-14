import { Link, NavLink, useLocation } from 'react-router-dom'
import { MaterialIcon } from './MaterialIcon'

export function TopHeader({
  title = 'Pipeline Explorer',
  subtitle,
}: {
  title?: string
  subtitle?: string
}) {
  const location = useLocation()
  const onStageRoute = location.pathname.startsWith('/stage/')
  return (
    <header className="sticky top-0 z-30 bg-surface flex justify-between items-center w-full px-8 py-3">
      <div className="flex items-center gap-8">
        <Link to="/" className="text-xl font-bold tracking-tight text-on-surface font-headline">
          {title}
        </Link>
        <nav className="hidden md:flex items-center gap-6 text-sm">
          <NavLink className={({ isActive }) => `${isActive && !onStageRoute ? 'text-primary font-semibold border-b-2 border-primary' : 'text-on-surface/60 hover:text-on-surface'} py-1`} to="/">
            Project narrative
          </NavLink>
          <NavLink
            className={({ isActive }) => `${isActive || onStageRoute ? 'text-primary font-semibold border-b-2 border-primary' : 'text-on-surface/60 hover:text-on-surface'} py-1`}
            to="/stage/residual"
          >
            Stage explorer
          </NavLink>
          <a
            className="text-on-surface/60 hover:text-on-surface py-1"
            href={import.meta.env.DEV ? '/pipeline_2026/stage2/s06_report/results/mmlu_report.html' : '#'}
            target="_blank"
            rel="noreferrer"
          >
            Report HTML
          </a>
        </nav>
      </div>
      {subtitle ? (
        <span className="text-xs text-on-surface-variant max-w-xs truncate hidden lg:inline">
          {subtitle}
        </span>
      ) : null}
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="p-2 text-on-surface-variant hover:bg-surface-container-low rounded-full"
          aria-label="Notifications"
        >
          <MaterialIcon name="notifications" />
        </button>
      </div>
    </header>
  )
}
