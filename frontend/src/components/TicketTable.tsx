import { useNavigate } from 'react-router-dom'
import type { Ticket } from '../types/helpdesk'
import RiskBadge from './RiskBadge'
import StatusBadge from './StatusBadge'
import {
  priorityLabel,
  queueLabel,
  typeLabel,
} from '../utils/labels'

const date = (value: string) =>
  new Intl.DateTimeFormat('ko-KR', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))

export default function TicketTable({ tickets }: { tickets: Ticket[] }) {
  const navigate = useNavigate()

  if (!tickets.length) {
    return (
      <div className='empty-state'>
        <span className='empty-icon'>◎</span>
        <h3>등록된 Ticket이 없습니다</h3>
        <p>새 Ticket을 등록하면 분석 workflow를 시작할 수 있습니다.</p>
      </div>
    )
  }

  return (
    <div className='table-wrap'>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>제목</th>
            <th>상태</th>
            <th>유형</th>
            <th>담당 큐</th>
            <th>우선순위</th>
            <th>위험도</th>
            <th>검토</th>
            <th>등록일</th>
          </tr>
        </thead>
        <tbody>
          {tickets.map((ticket) => (
            <tr
              key={ticket.id}
              onClick={() => navigate(`/tickets/${ticket.id}`)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  navigate(`/tickets/${ticket.id}`)
                }
              }}
              tabIndex={0}
            >
              <td className='mono'>#{ticket.id}</td>
              <td className='subject-cell'>{ticket.subject}</td>
              <td><StatusBadge status={ticket.status} /></td>
              <td>{typeLabel(ticket.predicted_type)}</td>
              <td>{queueLabel(ticket.predicted_queue)}</td>
              <td>{priorityLabel(ticket.predicted_priority)}</td>
              <td><RiskBadge risk={ticket.risk_level} /></td>
              <td>
                {ticket.review_required ? (
                  <span className='review-flag'>필요</span>
                ) : (
                  <span className='muted'>일반</span>
                )}
              </td>
              <td className='date-cell'>{date(ticket.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
