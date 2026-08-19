import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  analyzeTicket,
  approveTicket,
  getTicket,
  getTicketEvents,
  rejectTicket,
  submitForApproval,
} from '../api/helpdesk'
import AuditTimeline from '../components/AuditTimeline'
import RiskBadge from '../components/RiskBadge'
import SimilarTicketCard from '../components/SimilarTicketCard'
import StatusBadge from '../components/StatusBadge'
import type { Ticket, TicketEvent } from '../types/helpdesk'
import {
  priorityLabel,
  queueLabel,
  typeLabel,
} from '../utils/labels'

const percent = (value: number | null) =>
  value === null ? '—' : `${(value * 100).toFixed(1)}%`

const date = (value: string) =>
  new Intl.DateTimeFormat('ko-KR', {
    dateStyle: 'long',
    timeStyle: 'short',
  }).format(new Date(value))

export default function TicketDetailPage() {
  const { id } = useParams()
  const ticketId = Number(id)
  const [ticket, setTicket] = useState<Ticket | null>(null)
  const [events, setEvents] = useState<TicketEvent[]>([])
  const [draft, setDraft] = useState('')
  const [finalAnswer, setFinalAnswer] = useState('')
  const [rejectReason, setRejectReason] = useState('')
  const [showReject, setShowReject] = useState(false)
  const [loading, setLoading] = useState(true)
  const [action, setAction] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    if (!Number.isInteger(ticketId)) {
      setError('올바르지 않은 Ticket ID입니다.')
      setLoading(false)
      return
    }

    try {
      const [ticketData, eventData] = await Promise.all([
        getTicket(ticketId),
        getTicketEvents(ticketId),
      ])
      setTicket(ticketData)
      setEvents(eventData)
      setDraft(ticketData.draft_answer ?? '')
      setFinalAnswer(ticketData.final_answer ?? ticketData.draft_answer ?? '')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Ticket을 불러오지 못했습니다.')
    } finally {
      setLoading(false)
    }
  }, [ticketId])

  useEffect(() => {
    void load()
  }, [load])

  const runAction = async (
    name: string,
    operation: () => Promise<Ticket>,
  ) => {
    setAction(name)
    setError('')
    try {
      await operation()
      await load()
      setShowReject(false)
      setRejectReason('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '요청 처리 중 오류가 발생했습니다.')
    } finally {
      setAction('')
    }
  }

  if (loading) {
    return <div className='loading-state page-load'><span className='spinner' /> Ticket을 불러오는 중...</div>
  }

  if (!ticket) {
    return (
      <div className='page'>
        <div className='alert error'>{error || 'Ticket을 찾을 수 없습니다.'}</div>
        <Link className='button secondary' to='/'>대시보드로 돌아가기</Link>
      </div>
    )
  }

  const canAnalyze = ['PENDING', 'ANALYZED', 'REJECTED'].includes(ticket.status)
  const canSubmit = ['ANALYZED', 'REJECTED'].includes(ticket.status)
  const waiting = ticket.status === 'WAITING_APPROVAL'

  return (
    <div className='page'>
      <div className='breadcrumb'>
        <Link to='/'>대시보드</Link><span>/</span><strong>티켓 #{ticket.id}</strong>
      </div>

      <div className='detail-heading'>
        <div>
          <div className='heading-badges'>
            <span className='ticket-number'>#{ticket.id}</span>
            <StatusBadge status={ticket.status} />
            <RiskBadge risk={ticket.risk_level} />
          </div>
          <h1>{ticket.subject}</h1>
          <p>등록 {date(ticket.created_at)}</p>
        </div>
        {canAnalyze && (
          <button
            className='button primary'
            disabled={Boolean(action)}
            onClick={() => runAction('analyze', () => analyzeTicket(ticket.id))}
          >
            {action === 'analyze' ? '분석 중...' : ticket.status === 'PENDING' ? 'AI 분석' : '재분석'}
          </button>
        )}
      </div>

      {error && <div className='alert error' role='alert'>{error}</div>}

      <section className='detail-grid'>
        <article className='panel detail-card original-card'>
          <div className='panel-head'>
            <div><p className='eyebrow'>원본 문의</p><h2>티켓 상세</h2></div>
          </div>
          <dl className='ticket-original'>
            <div><dt>제목</dt><dd>{ticket.subject}</dd></div>
            <div><dt>문의 내용</dt><dd className='body-copy'>{ticket.body}</dd></div>
            <div><dt>상태</dt><dd><StatusBadge status={ticket.status} /></dd></div>
            <div><dt>등록일</dt><dd>{date(ticket.created_at)}</dd></div>
          </dl>
        </article>

        <article className='panel detail-card analysis-card'>
          <div className='panel-head'>
            <div><p className='eyebrow'>AI 분석</p><h2>분류 및 위험도</h2></div>
            {ticket.analyzed_at && <span className='count-label'>{date(ticket.analyzed_at)}</span>}
          </div>
          {ticket.predicted_type ? (
            <>
              <div className='prediction-grid'>
                <div><span>유형</span><strong>{typeLabel(ticket.predicted_type)}</strong><em>{percent(ticket.type_confidence)}</em></div>
                <div><span>담당 큐</span><strong>{queueLabel(ticket.predicted_queue)}</strong><em>{percent(ticket.queue_confidence)}</em></div>
                <div><span>우선순위</span><strong>{priorityLabel(ticket.predicted_priority)}</strong><em>{percent(ticket.priority_confidence)}</em></div>
              </div>
              <div className='risk-summary'>
                <div><span>검색 Top1 유사도</span><strong>{ticket.retrieval_top1_similarity?.toFixed(4) ?? '—'}</strong></div>
                <div><span>위험도</span><RiskBadge risk={ticket.risk_level} /></div>
                <div><span>담당자 검토</span><strong>{ticket.review_required ? '강화 검토 필요' : '일반 승인'}</strong></div>
              </div>
              <div className='risk-reasons'>
                <span>위험 사유</span>
                {ticket.risk_reasons?.length ? (
                  ticket.risk_reasons.map((reason) => <code key={reason}>{reason}</code>)
                  ) : <strong>위험 사유 없음</strong>}
              </div>
            </>
          ) : (
            <div className='empty-inline'><span>◇</span><p>아직 AI 분석을 실행하지 않았습니다.</p></div>
          )}
        </article>
      </section>

      {ticket.similar_tickets && ticket.similar_tickets.length > 0 && (
        <section className='section-block'>
          <div className='section-heading'>
            <div><p className='eyebrow'>지식 검색</p><h2>유사 과거 티켓</h2></div>
            <span className='count-label'>Top {ticket.similar_tickets.length}</span>
          </div>
          <div className='similar-list'>
            {ticket.similar_tickets.slice(0, 3).map((item, index) => (
              <SimilarTicketCard key={`${item.subject}-${index}`} ticket={item} rank={index + 1} />
            ))}
          </div>
        </section>
      )}

      {(canSubmit || waiting || ticket.status === 'APPROVED') && (
        <section className='panel review-panel'>
          <div className='panel-head'>
            <div><p className='eyebrow'>담당자 검토</p><h2>{waiting ? '승인 결정' : ticket.status === 'APPROVED' ? '승인된 답변' : '답변 작성'}</h2></div>
            {ticket.review_required && <span className='review-flag'>강화 검토 필요</span>}
          </div>

          {canSubmit && (
            <>
              {ticket.status === 'REJECTED' && ticket.review_comment && (
                <div className='alert warning'><strong>이전 반려 사유:</strong> {ticket.review_comment}</div>
              )}
              <label><span>답변 초안</span><textarea rows={8} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder='고객에게 전달할 답변 초안을 입력하세요.' /></label>
              <div className='form-actions'>
                <button className='button primary' disabled={Boolean(action)} onClick={() => runAction('submit', () => submitForApproval(ticket.id, { draft_answer: draft || null }))}>
                  {action === 'submit' ? '요청 중...' : '승인 요청'}
                </button>
              </div>
            </>
          )}

          {waiting && (
            <>
              <div className='draft-preview'><span>제출된 초안</span><p>{ticket.draft_answer || '작성된 답변 초안이 없습니다.'}</p></div>
              <label><span>최종 답변</span><textarea rows={8} value={finalAnswer} onChange={(event) => setFinalAnswer(event.target.value)} placeholder='승인할 최종 답변을 검토·수정하세요.' /></label>
              {showReject && <label><span>반려 사유</span><textarea rows={4} value={rejectReason} onChange={(event) => setRejectReason(event.target.value)} placeholder='반려 사유를 구체적으로 입력하세요.' required /></label>}
              <div className='form-actions split-actions'>
                <button className='button danger-outline' onClick={() => setShowReject((value) => !value)}>반려</button>
                {showReject && <button className='button danger' disabled={!rejectReason.trim() || Boolean(action)} onClick={() => runAction('reject', () => rejectTicket(ticket.id, { reason: rejectReason }))}>{action === 'reject' ? '반려 중...' : '반려 확정'}</button>}
                <button className='button success' disabled={Boolean(action)} onClick={() => runAction('approve', () => approveTicket(ticket.id, { final_answer: finalAnswer || null }))}>{action === 'approve' ? '승인 중...' : '답변 승인'}</button>
              </div>
            </>
          )}

          {ticket.status === 'APPROVED' && <div className='approved-answer'><span>최종 답변</span><p>{ticket.final_answer || '최종 답변이 비어 있습니다.'}</p><strong>✓ 담당자 승인 완료</strong></div>}
        </section>
      )}

      <section className='panel audit-panel'>
        <div className='panel-head'>
          <div><p className='eyebrow'>추적성</p><h2>처리 이력</h2></div>
          <span className='count-label'>{events.length}개 이벤트</span>
        </div>
        <AuditTimeline events={events} />
      </section>
    </div>
  )
}
