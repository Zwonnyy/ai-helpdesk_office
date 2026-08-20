export type TicketStatus =
  | 'PENDING'
  | 'ANALYZED'
  | 'WAITING_APPROVAL'
  | 'APPROVED'
  | 'REJECTED'

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH'

export interface PredictionResult {
  label: string
  confidence: number
}

export interface SimilarTicket {
  kb_index?: number | null
  score: number
  subject: string
  body: string
  answer: string
  type: string
  queue: string
  priority: string
  language: string
  subject_ko?: string | null
  body_ko?: string | null
  answer_ko?: string | null
  translation_status?: string | null
}

export interface SimilarTicketTranslation {
  kb_index: number | null
  target_language: string
  subject: string
  body: string
  answer: string
  cached: boolean
  translated: boolean
  error: string | null
}

export interface Ticket {
  id: number
  subject: string
  body: string
  status: TicketStatus
  predicted_type: string | null
  type_confidence: number | null
  predicted_queue: string | null
  queue_confidence: number | null
  predicted_priority: string | null
  priority_confidence: number | null
  similar_tickets: SimilarTicket[] | null
  retrieval_top1_similarity: number | null
  risk_level: RiskLevel | null
  review_required: boolean
  risk_reasons: string[] | null
  draft_answer: string | null
  final_answer: string | null
  review_comment: string | null
  created_at: string
  updated_at: string
  analyzed_at: string | null
  reviewed_at: string | null
}

export interface TicketEventData {
  type?: PredictionResult
  queue?: PredictionResult
  priority?: PredictionResult
  retrieved_count?: number
  risk_level?: RiskLevel
  review_required?: boolean
  reasons?: string[]
  threshold_inputs?: {
    type_confidence?: number
    queue_confidence?: number
    priority_confidence?: number
    retrieval_similarity?: number
  }
  has_draft_answer?: boolean
  final_answer_present?: boolean
  reason?: string
}

export interface TicketEvent {
  id: number
  ticket_id: number
  event_type: string
  from_status: string | null
  to_status: string | null
  message: string | null
  event_data: TicketEventData | null
  created_at: string
}

export interface CreateTicketInput {
  subject: string
  body: string
}

export interface SubmitTicketInput {
  draft_answer: string | null
}

export interface ApproveTicketInput {
  final_answer: string | null
}

export interface RejectTicketInput {
  reason: string
}
