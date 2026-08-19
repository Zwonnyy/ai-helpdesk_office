import { useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { createTicket } from '../api/helpdesk'

export default function NewTicketPage() {
  const navigate = useNavigate()
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setSubmitting(true)
    setError('')

    try {
      const ticket = await createTicket({ subject, body })
      navigate(`/tickets/${ticket.id}`)
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : 'Ticket 생성 중 오류가 발생했습니다.',
      )
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className='page narrow-page'>
      <div className='breadcrumb'>
        <Link to='/'>대시보드</Link><span>/</span><strong>새 티켓</strong>
      </div>
      <div className='page-heading'>
        <div>
          <p className='eyebrow'>티켓 접수</p>
          <h1>새 지원 티켓</h1>
          <p>고객의 문의 원문을 등록합니다. 생성 후 AI 분석을 실행할 수 있습니다.</p>
        </div>
      </div>

      <form className='panel ticket-form' onSubmit={handleSubmit}>
        {error && <div className='alert error' role='alert'>{error}</div>}
        <label>
          <span>제목</span>
          <input
            value={subject}
            onChange={(event) => setSubject(event.target.value)}
            placeholder='예: VPN 연결 시 인증 오류가 발생합니다'
            required
            autoFocus
          />
          <small>Ticket을 빠르게 식별할 수 있는 제목을 입력하세요.</small>
        </label>
        <label>
          <span>문의 내용</span>
          <textarea
            value={body}
            onChange={(event) => setBody(event.target.value)}
            placeholder='발생한 문제, 오류 메시지, 이미 시도한 조치를 자세히 입력하세요.'
            rows={12}
            required
          />
          <small>AI classification과 유사 사례 검색에 사용됩니다.</small>
        </label>
        <div className='form-actions'>
          <Link className='button secondary' to='/'>취소</Link>
          <button className='button primary' disabled={submitting}>
            {submitting ? '등록 중...' : '티켓 등록'}
          </button>
        </div>
      </form>
    </div>
  )
}
