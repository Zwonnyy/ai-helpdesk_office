const typeLabels: Record<string, string> = {
  Incident: '장애',
  Request: '요청',
  Problem: '문제',
  Change: '변경',
}

const priorityLabels: Record<string, string> = {
  high: '높음',
  medium: '보통',
  low: '낮음',
}

const queueLabels: Record<string, string> = {
  'Billing and Payments': '청구 및 결제',
  'Customer Service': '고객 서비스',
  'General Inquiry': '일반 문의',
  'Human Resources': '인사',
  'IT Support': 'IT 지원',
  'Product Support': '제품 지원',
  'Returns and Exchanges': '반품 및 교환',
  'Sales and Pre-Sales': '영업 및 사전 영업',
  'Service Outages and Maintenance': '서비스 장애 및 유지보수',
  'Technical Support': '기술 지원',
}

const statusLabels: Record<string, string> = {
  PENDING: '대기',
  ANALYZED: '분석 완료',
  WAITING_APPROVAL: '승인 대기',
  APPROVED: '승인 완료',
  REJECTED: '반려',
}

const riskLabels: Record<string, string> = {
  LOW: '낮음',
  MEDIUM: '주의',
  HIGH: '높음',
}

const riskReasonLabels: Record<string, string> = {
  LOW_QUEUE_CONFIDENCE: 'Queue 신뢰도가 기준보다 낮습니다.',
  LOW_PRIORITY_CONFIDENCE: 'Priority 신뢰도가 기준보다 낮습니다.',
  LOW_TYPE_CONFIDENCE: 'Type 신뢰도가 기준보다 낮습니다.',
  LOW_RETRIEVAL_SIMILARITY: '유사 사례 검색 신뢰도가 기준보다 낮습니다.',
}

export const typeLabel = (value: string | null) =>
  value ? (typeLabels[value] ?? value) : '—'

export const priorityLabel = (value: string | null) =>
  value ? (priorityLabels[value.toLowerCase()] ?? value) : '—'

export const queueLabel = (value: string | null) =>
  value ? (queueLabels[value] ?? value) : '—'

export const statusLabel = (value: string | null) =>
  value ? (statusLabels[value] ?? value) : '신규'

export const riskLabel = (value: string | null) =>
  value ? (riskLabels[value] ?? value) : '미평가'

export const riskReasonLabel = (value: string) =>
  riskReasonLabels[value] ?? value
