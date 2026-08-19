import type {
  ApproveTicketInput,
  CreateTicketInput,
  RejectTicketInput,
  SubmitTicketInput,
  Ticket,
  TicketEvent,
} from '../types/helpdesk'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    let message = `요청에 실패했습니다. (${response.status})`

    try {
      const error = (await response.json()) as { detail?: string }
      if (error.detail) message = error.detail
    } catch {
      // Keep the status-based fallback.
    }

    throw new ApiError(message, response.status)
  }

  return response.json() as Promise<T>
}

export const getTickets = () => request<Ticket[]>('/tickets')

export const getTicket = (id: number) =>
  request<Ticket>(`/tickets/${id}`)

export const createTicket = (data: CreateTicketInput) =>
  request<Ticket>('/tickets', {
    method: 'POST',
    body: JSON.stringify(data),
  })

export const analyzeTicket = (id: number) =>
  request<Ticket>(`/tickets/${id}/analyze`, {
    method: 'POST',
  })

export const submitForApproval = (
  id: number,
  data: SubmitTicketInput,
) =>
  request<Ticket>(`/tickets/${id}/submit-for-approval`, {
    method: 'POST',
    body: JSON.stringify(data),
  })

export const approveTicket = (
  id: number,
  data: ApproveTicketInput,
) =>
  request<Ticket>(`/tickets/${id}/approve`, {
    method: 'POST',
    body: JSON.stringify(data),
  })

export const rejectTicket = (
  id: number,
  data: RejectTicketInput,
) =>
  request<Ticket>(`/tickets/${id}/reject`, {
    method: 'POST',
    body: JSON.stringify(data),
  })

export const getTicketEvents = (id: number) =>
  request<TicketEvent[]>(`/tickets/${id}/events`)
