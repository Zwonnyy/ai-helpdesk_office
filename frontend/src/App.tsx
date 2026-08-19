import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import DashboardPage from './pages/DashboardPage'
import NewTicketPage from './pages/NewTicketPage'
import TicketDetailPage from './pages/TicketDetailPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path='tickets/new' element={<NewTicketPage />} />
          <Route path='tickets/:id' element={<TicketDetailPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
