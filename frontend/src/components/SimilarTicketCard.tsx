import { useState } from 'react'
import type { SimilarTicket } from '../types/helpdesk'
import {
  priorityLabel,
  queueLabel,
  typeLabel,
} from '../utils/labels'

export default function SimilarTicketCard({
  ticket,
  rank,
}: {
  ticket: SimilarTicket
  rank: number
}) {
  const [expanded, setExpanded] = useState(false)
  const [showOriginal, setShowOriginal] = useState(false)
  const localized = ticket.translation_status === 'completed'
  const subject = (
    localized && !showOriginal
      ? ticket.subject_ko
      : ticket.subject
  ) || ticket.subject
  const body = (
    localized && !showOriginal
      ? ticket.body_ko
      : ticket.body
  ) || ticket.body
  const answer = (
    localized && !showOriginal
      ? ticket.answer_ko
      : ticket.answer
  ) || ticket.answer

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
        {localized && (
          <button
            className='text-button original-toggle'
            type='button'
            onClick={() => setShowOriginal((value) => !value)}
          >
            {showOriginal ? '한국어 번역 보기' : '원문 보기'}
          </button>
        )}
      </div>
    </article>
  )
}
