import type { TicketEvent, TicketEventData } from '../types/helpdesk'
import {
  priorityLabel,
  queueLabel,
  riskLabel,
  statusLabel,
  typeLabel,
} from '../utils/labels'

const eventLabels: Record<string, string> = {
  TICKET_CREATED: '티켓 생성',
  AI_ANALYZED: 'AI 분석',
  RISK_EVALUATED: '위험도 평가',
  SUBMITTED_FOR_APPROVAL: '승인 요청',
  APPROVED: '승인 완료',
  REJECTED: '반려',
}

const eventMessages: Record<string, string> = {
  TICKET_CREATED: '티켓이 생성되었습니다.',
  AI_ANALYZED: 'AI 분류와 유사 사례 검색이 완료되었습니다.',
  RISK_EVALUATED: 'AI 검토 위험도를 평가했습니다.',
  SUBMITTED_FOR_APPROVAL: '담당자 승인 요청을 제출했습니다.',
  APPROVED: '담당자가 답변을 승인했습니다.',
  REJECTED: '담당자가 답변을 반려했습니다.',
}

const date = (value: string) =>
  new Intl.DateTimeFormat('ko-KR', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(new Date(value))

function EventDetails({
  type,
  data,
}: {
  type: string
  data: TicketEventData | null
}) {
  if (!data) return null

  if (type === 'AI_ANALYZED') {
    return (
      <dl className='event-details'>
        <div><dt>유형</dt><dd>{typeLabel(data.type?.label ?? null)}</dd></div>
        <div><dt>담당 큐</dt><dd>{queueLabel(data.queue?.label ?? null)}</dd></div>
        <div><dt>우선순위</dt><dd>{priorityLabel(data.priority?.label ?? null)}</dd></div>
        <div><dt>검색 결과</dt><dd>{data.retrieved_count ?? 0}</dd></div>
      </dl>
    )
  }

  if (type === 'RISK_EVALUATED') {
    return (
      <dl className='event-details'>
        <div><dt>위험도</dt><dd>{riskLabel(data.risk_level ?? null)}</dd></div>
        <div>
          <dt>검토</dt>
          <dd>{data.review_required ? '강화 검토' : '일반 승인'}</dd>
        </div>
        <div>
          <dt>유사도</dt>
          <dd>
            {data.threshold_inputs?.retrieval_similarity?.toFixed(4) ?? '—'}
          </dd>
        </div>
        <div className='wide'>
          <dt>사유</dt>
          <dd>{data.reasons?.join(', ') || '위험 사유 없음'}</dd>
        </div>
      </dl>
    )
  }

  if (type === 'REJECTED' && data.reason) {
    return <p className='event-note'>반려 사유: {data.reason}</p>
  }

  return null
}

export default function AuditTimeline({
  events,
}: {
  events: TicketEvent[]
}) {
  if (!events.length) {
    return <p className='muted'>기록된 처리 이력이 없습니다.</p>
  }

  return (
    <ol className='timeline'>
      {events.map((event) => (
        <li key={event.id}>
          <span className='timeline-dot' />
          <div className='timeline-head'>
            <div>
              <strong>{eventLabels[event.event_type] ?? event.event_type}</strong>
              {event.from_status !== event.to_status && (
                <span className='transition'>
                  {statusLabel(event.from_status)} → {statusLabel(event.to_status)}
                </span>
              )}
            </div>
            <time>{date(event.created_at)}</time>
          </div>
          {(eventMessages[event.event_type] || event.message) && (
            <p>{eventMessages[event.event_type] ?? event.message}</p>
          )}
          <EventDetails type={event.event_type} data={event.event_data} />
        </li>
      ))}
    </ol>
  )
}
