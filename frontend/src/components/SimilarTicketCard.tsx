import { useState } from 'react'
import type {
  SimilarTicket,
  SimilarTicketTranslation,
} from '../types/helpdesk'
import {
  priorityLabel,
  queueLabel,
  typeLabel,
} from '../utils/labels'

export default function SimilarTicketCard({
  ticket,
  rank,
  translation,
  showTranslation,
}: {
  ticket: SimilarTicket
  rank: number
  translation?: SimilarTicketTranslation
  showTranslation: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  const localized = showTranslation && translation?.translated
  const subject = localized ? translation.subject : ticket.subject
  const body = localized ? translation.body : ticket.body
  const answer = localized ? translation.answer : ticket.answer

  return (
    <article className='similar-card'>
      <div className='similar-head'>
        <span className='rank'>#{rank}</span>
        <div>
          <p className='eyebrow'>유사 사례</p>
          <h3>{subject || '제목 없음'}</h3>
        </div>
        <strong className='similarity'>
          {(ticket.score * 100).toFixed(1)}%
          <span>유사도</span>
        </strong>
      </div>
      <div className='metadata-row'>
        <span>{typeLabel(ticket.type)}</span>
        <span>{queueLabel(ticket.queue)}</span>
        <span>{priorityLabel(ticket.priority)}</span>
        {localized && <span>{translation.cached ? '한국어 번역 · 캐시됨' : '한국어 번역'}</span>}
      </div>
      <div className='historical-answer'>
        <p className='field-label'>문의 내용</p>
        <p className={expanded ? '' : 'line-clamp'}>{body}</p>
        <p className='field-label answer-label'>기존 답변</p>
        <p className={expanded ? '' : 'line-clamp'}>{answer}</p>
        {(body.length + answer.length) > 220 && (
          <button
            className='text-button'
            type='button'
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? '접기' : '전체 내용 보기'}
          </button>
        )}
        {showTranslation && translation && !translation.translated && (
          <p className='translation-fallback'>번역에 실패했습니다. 원문을 표시합니다.</p>
        )}
      </div>
    </article>
  )
}
