import type { TicketStatus } from '../types/helpdesk'

const labels: Record<TicketStatus, string> = {
  PENDING: '대기',
  ANALYZED: '분석 완료',
  WAITING_APPROVAL: '승인 대기',
  APPROVED: '승인 완료',
  REJECTED: '반려',
}

export default function StatusBadge({ status }: { status: TicketStatus }) {
  return (
    <span className={`badge status status-${status.toLowerCase()}`}>
      {labels[status]}
    </span>
  )
}
