import { Navigate, Route, Routes } from 'react-router-dom'
import { Landing } from './pages/Landing'
import { Poster, PosterEN, PosterID } from './pages/Poster'
import { StageLayout } from './pages/StageLayout'
import {
  DiffStage,
  EvalStage,
  ExportStage,
  ReportStage,
  ResidualStage,
  SaeStage,
} from './pages/stages'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/poster" element={<Poster />} />
      <Route path="/poster-en" element={<PosterEN />} />
      <Route path="/poster-id" element={<PosterID />} />
      <Route element={<StageLayout />}>
        <Route path="stage/residual" element={<ResidualStage />} />
        <Route path="stage/export" element={<ExportStage />} />
        <Route path="stage/sae" element={<SaeStage />} />
        <Route path="stage/diff" element={<DiffStage />} />
        <Route path="stage/eval" element={<EvalStage />} />
        <Route path="stage/report" element={<ReportStage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
