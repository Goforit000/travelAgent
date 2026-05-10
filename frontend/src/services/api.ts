import axios from 'axios'
import type { TripFormData, TripPlanResponse } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// ============================================================
// 请求拦截器
// ============================================================

apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// ============================================================
// 响应拦截器
// ============================================================

apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

// ============================================================
// 普通 JSON 请求：生成旅行计划（兼容旧版）
// ============================================================

export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

// ============================================================
// SSE 事件类型定义
// ============================================================

export interface SSEEvent {
  type: 'agent_start' | 'agent_end' | 'tool_start' | 'tool_end' | 'progress' | 'done' | 'error'
  data: any
}

export interface StreamCallbacks {
  onAgentStart?: (agentName: string, raw: any) => void
  onAgentEnd?: (agentName: string, raw: any) => void
  onToolStart?: (toolName: string, agentName: string, raw: any) => void
  onToolEnd?: (toolName: string, agentName: string, raw: any) => void
  onProgress?: (percent: number, message: string) => void
  onDone?: (data: any, message: string) => void
  onError?: (message: string) => void
}

// ============================================================
// SSE 流式请求：生成旅行计划（真实进度推送）
// ============================================================

export async function generateTripStream(
  formData: TripFormData,
  callbacks: StreamCallbacks
): Promise<void> {
  const { onAgentStart, onAgentEnd, onToolStart, onToolEnd, onProgress, onDone, onError } = callbacks

  try {
    const response = await fetch(`${API_BASE_URL}/api/trip/plan/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(formData),
    })

    if (!response.ok) {
      const errorText = await response.text()
      onError?.(`请求失败 (${response.status}): ${errorText}`)
      return
    }

    const contentType = response.headers.get('content-type') || ''
    if (!contentType.includes('text/event-stream')) {
      // 非 SSE 响应，尝试作为 JSON 解析
      const json = await response.json()
      if (json.success && json.data) {
        onDone?.(json.data, json.message || '')
      } else {
        onError?.(json.message || '未知错误')
      }
      return
    }

    // 解析 SSE 流
    const reader = response.body?.getReader()
    if (!reader) {
      onError?.('无法读取响应流')
      return
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })

      // 按行解析 SSE 数据
      const lines = buffer.split('\n')
      // 最后一个可能是不完整的行，保留到下次
      buffer = lines.pop() || ''

      let currentEvent = ''
      let currentData = ''

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          currentData = line.slice(6)
        } else if (line === '' && currentEvent && currentData) {
          // 空行 = 一个事件结束，分发回调
          try {
            const parsedData = JSON.parse(currentData)
            _dispatchEvent(currentEvent, parsedData, {
              onAgentStart,
              onAgentEnd,
              onToolStart,
              onToolEnd,
              onProgress,
              onDone,
              onError,
            })
          } catch {
            // JSON 解析失败，忽略
            console.warn('SSE JSON 解析失败:', currentData)
          }

          currentEvent = ''
          currentData = ''
        }
      }

      // 处理最后可能残留的完整事件（无尾部空行）
      if (currentEvent && currentData) {
        try {
          const parsedData = JSON.parse(currentData)
          _dispatchEvent(currentEvent, parsedData, {
            onAgentStart,
            onAgentEnd,
            onToolStart,
            onToolEnd,
            onProgress,
            onDone,
            onError,
          })
        } catch {
          // 忽略
        }
      }
    }
  } catch (error: any) {
    console.error('SSE 流请求失败:', error)
    onError?.(error.message || '流式连接失败')
  }
}

/**
 * 根据事件类型分发到对应的回调
 */
function _dispatchEvent(
  eventType: string,
  data: any,
  callbacks: StreamCallbacks
): void {
  switch (eventType) {
    case 'agent_start':
      callbacks.onAgentStart?.(data.name || data.agent, data)
      break
    case 'agent_end':
      callbacks.onAgentEnd?.(data.name || data.agent, data)
      break
    case 'tool_start':
      callbacks.onToolStart?.(data.tool, data.agent, data)
      break
    case 'tool_end':
      callbacks.onToolEnd?.(data.tool, data.agent, data)
      break
    case 'progress':
      callbacks.onProgress?.(data.percent || 0, data.message || '')
      break
    case 'done':
      callbacks.onDone?.(data.data, data.message || '')
      break
    case 'error':
      callbacks.onError?.(data.message || '未知错误')
      break
    default:
      console.log('未处理的 SSE 事件:', eventType, data)
  }
}

// ============================================================
// 历史计划
// ============================================================

export async function fetchTripHistory(limit: number = 20): Promise<any[]> {
  try {
    const response = await apiClient.get('/api/history', { params: { limit } })
    return response.data?.data || []
  } catch (error: any) {
    console.error('获取历史记录失败:', error)
    return []
  }
}

export async function fetchTripDetail(tripId: string): Promise<any> {
  try {
    const response = await apiClient.get(`/api/history/${tripId}`)
    return response.data?.data || null
  } catch (error: any) {
    console.error('获取计划详情失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '获取失败')
  }
}

export async function deleteTrip(tripId: string): Promise<void> {
  try {
    await apiClient.delete(`/api/history/${tripId}`)
  } catch (error: any) {
    console.error('删除计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '删除失败')
  }
}

// ============================================================
// 健康检查
// ============================================================

export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient
