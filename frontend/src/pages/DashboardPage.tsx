import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { getTickets } from '../api/helpdesk'
import TicketTable from '../components/TicketTable'
import type { Ticket } from '../types/helpdesk'

export default function DashboardPage() {
  const [tickets, setTickets] = useState<Ticket[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getTickets()
      .then(setTickets)
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false))
  }, [])

  const counts = useMemo(() => {
    const status = {
      PENDING: 0,
      ANALYZED: 0,
      WAITING_APPROVAL: 0,
      APPROVED: 0,
      REJECTED: 0,
    }
    const risk = { LOW: 0, MEDIUM: 0, HIGH: 0 }

    tickets.forEach((ticket) => {
      if (ticket.status in status) {
        status[ticket.status as keyof typeof status] += 1
      }
      if (ticket.risk_level) risk[ticket.risk_level] += 1
    })

    return { status, risk }
  }, [tickets])

  return (
    <div className='page'>
      <div className='page-heading'>
        <div>
          <p className='eyebrow'>운영 현황</p>
          <h1>티켓 대시보드</h1>
          <p>AI 분석 상태와 담당자 검토 대기열을 한 곳에서 확인합니다.</p>
        </div>
        <Link className='button primary' to='/tickets/new'>
          + 티켓 등록
        </Link>
      </div>

      {error && <div className='alert error' role='alert'>{error}</div>}

      <section className='summary-grid' aria-label='Ticket 현황'>
        <article className='summary-card accent'>
          <span>전체 티켓</span>
          <strong>{tickets.length}</strong>
          <small>전체 등록 건</small>
        </article>
        <article className='summary-card'>
          <span>대기</span>
          <strong>{counts.status.PENDING}</strong>
          <small>분석 대기</small>
        </article>
        <article className='summary-card analyzed'>
          <span>분석 완료</span>
          <strong>{counts.status.ANALYZED}</strong>
          <small>답변 작성 가능</small>
        </article>
        <article className='summary-card attention'>
          <span>승인 대기</span>
          <strong>{counts.status.WAITING_APPROVAL}</strong>
          <small>검토 필요</small>
        </article>
        <article className='summary-card success'>
          <span>승인 완료</span>
          <strong>{counts.status.APPROVED}</strong>
          <small>처리 완료</small>
        </article>
        <article className='summary-card danger'>
          <span>반려</span>
          <strong>{counts.status.REJECTED}</strong>
          <small>수정 필요</small>
        </article>
      </section>

      <section className='risk-strip' aria-label='위험도 현황'>
        <span className='section-label'>위험도 분포</span>
        <div><i className='risk-low-bg' /> 낮음 <strong>{counts.risk.LOW}</strong></div>
        <div><i className='risk-medium-bg' /> 주의 <strong>{counts.risk.MEDIUM}</strong></div>
        <div><i className='risk-high-bg' /> 높음 <strong>{counts.risk.HIGH}</strong></div>
      </section>

      <section className='panel'>
        <div className='panel-head'>
          <div>
            <h2>최근 티켓</h2>
            <p>Ticket을 선택하면 분석과 승인 workflow를 진행할 수 있습니다.</p>
          </div>
          <span className='count-label'>{tickets.length}건</span>
        </div>
        {loading ? (
          <div className='loading-state'><span className='spinner' /> Ticket을 불러오는 중...</div>
        ) : (
          <TicketTable tickets={tickets} />
        )}
      </section>
    </div>
  )
}
