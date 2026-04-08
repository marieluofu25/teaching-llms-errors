import { Outlet } from 'react-router-dom'
import { StageSidebar } from '../components/StageSidebar'
import { TopHeader } from '../components/TopHeader'

export function StageLayout() {
  return (
    <div className="min-h-screen bg-surface">
      <StageSidebar />
      <div className="pl-64 flex flex-col min-h-screen">
        <TopHeader />
        <main className="flex-1 p-8 max-w-[1600px]">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
