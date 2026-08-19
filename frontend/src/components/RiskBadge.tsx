import type { RiskLevel } from '../types/helpdesk'

export default function RiskBadge({
  risk,
}: {
  risk: RiskLevel | null
}) {
  if (!risk) return <span className='muted'>미평가</span>

  const label = {
    LOW: '낮음',
    MEDIUM: '주의',
    HIGH: '높음',
  }[risk]
  return (
    <span className={`badge risk risk-${risk.toLowerCase()}`}>
      <span className='badge-dot' aria-hidden='true' />
      위험도 {label}
    </span>
  )
}
