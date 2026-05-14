// ============================================================
// 第一层：基础类型
// ============================================================

export interface Location {
  longitude: number
  latitude: number
}

// ============================================================
// 第二层：业务实体
// ============================================================

export interface Attraction {
  name: string
  address: string
  location: Location
  visit_duration: number
  description: string
  category?: string
  rating?: number
  image_url?: string
  ticket_price?: number
}

export interface Hotel {
  name: string
  address: string
  location?: Location
  price_range: string
  rating: number
  distance: string
  type: string
  estimated_cost?: number
}

export interface Meal {
  type: 'breakfast' | 'lunch' | 'dinner' | 'snack'
  name: string
  address?: string
  location?: Location
  description?: string
  estimated_cost?: number
}

export interface WeatherInfo {
  date: string
  day_weather: string
  night_weather: string
  day_temp: number
  night_temp: number
  wind_direction: string
  wind_power: string
}

export interface IndoorBackupAttraction {
  name: string
  address: string
  description: string
  category: string
  reason: string
  estimated_duration: number
  ticket_price: number
}

export interface Budget {
  total_attractions: number
  total_hotels: number
  total_meals: number
  total_transportation: number
  total_intercity_transport?: number
  total: number
}

/** 预算明细 — Budget Agent 输出的完整分析 */
export interface BudgetDetail {
  total_attractions: number
  total_hotels: number
  total_meals: number
  total_transportation: number
  total_intercity_transport?: number
  total: number
  daily_breakdown?: Array<Record<string, any>>
  savings_suggestions?: Array<Record<string, any>>
  warnings?: Array<Record<string, any>>
  analysis?: string
}

// ============================================================
// 第三层：组合类型
// ============================================================

export interface DayPlan {
  date: string
  day_index: number
  description: string
  transportation: string
  accommodation: string
  hotel?: Hotel
  attractions: Attraction[]
  meals: Meal[]
}

export type IntercityTransportMode = 'driving' | 'high_speed_rail' | 'flight'

export interface IntercityRouteSegment {
  direction: 'outbound' | 'return'
  origin: string
  destination: string
  date: string
  mode: IntercityTransportMode
  duration_minutes: number
  distance_km: number
  estimated_cost: number
  route_summary: string
  notes: string[]
  service_no?: string | null
  carrier?: string | null
  departure_place?: string | null
  arrival_place?: string | null
  departure_time?: string | null
  arrival_time?: string | null
  price_per_person?: number | null
  data_source?: 'amap_railway' | 'amap_flight' | 'amap_driving' | 'estimated' | string
  is_estimated?: boolean
}

export interface IntercityTransportPlan {
  mode: IntercityTransportMode
  outbound: IntercityRouteSegment
  return_trip: IntercityRouteSegment
  total_cost: number
  total_duration_minutes: number
  summary: string
  warnings: string[]
}

export interface TripPlan {
  city: string
  departure_city?: string | null
  start_date: string
  end_date: string
  people_count?: number
  indoor_backup_attractions?: IndoorBackupAttraction[]
  intercity_transport_mode?: IntercityTransportMode
  intercity_transport?: IntercityTransportPlan | null
  days: DayPlan[]
  weather_info: WeatherInfo[]
  overall_suggestions: string
  budget?: Budget
  /** Budget Agent 计算的总预算金额 */
  total_budget?: number
  /** Budget Agent 输出的完整预算分析 */
  budget_details?: BudgetDetail
  /** 工作流元数据（调试用） */
  workflow_metadata?: Record<string, any>
}

// ============================================================
// 第四层：请求 & 响应
// ============================================================

export interface TripFormData {
  departure_city?: string
  city: string
  start_date: string
  end_date: string
  travel_days: number
  intercity_transport_mode: IntercityTransportMode
  transportation: string
  accommodation: string
  preferences: string[]
  free_text_input: string
  target_budget: number
  people_count: number
}

export interface TripPlanResponse {
  success: boolean
  message: string
  data?: TripPlan
  trip_id?: string
}

/** 历史计划列表项 */
export interface HistoryItem {
  id: string
  departure_city?: string | null
  city: string
  start_date: string
  end_date: string
  travel_days: number
  people_count: number
  intercity_transport_mode?: IntercityTransportMode
  preferences: string[]
  budget_total: number | null
  created_at: string
}

// ============================================================
// SSE 事件类型（前后端实时通信）
// ============================================================

/** SSE 事件的 type 字段字面量 */
export type SSEEventType =
  | 'agent_start'
  | 'agent_end'
  | 'tool_start'
  | 'tool_end'
  | 'progress'
  | 'done'
  | 'error'

/** SSE agent_start / agent_end 事件的 data */
export interface SSEAgentData {
  agent: string   // 节点内部名称，如 "poi_node"
  name: string    // 用户可见名称，如 "景点搜索"
}

/** SSE tool_start / tool_end 事件的 data */
export interface SSEToolData {
  tool: string    // 工具名称
  agent: string   // 所属节点的内部名称
}

/** SSE progress 事件的 data */
export interface SSEProgressData {
  percent: number   // 进度百分比 0-100
  node: string      // 当前节点名称
  name: string      // 用户可见的阶段名称
  message: string   // 进度消息
}

/** SSE done 事件的 data */
export interface SSEDoneData {
  success: boolean
  message: string
  data: TripPlan
}

/** SSE error 事件的 data */
export interface SSEErrorData {
  success: boolean
  message: string
}

/** SSE 事件联合类型 */
export interface ProgressEvent {
  type: SSEEventType
  agent?: string
  name?: string
  tool?: string
  message?: string
  percent?: number
  percentage?: number
  node?: string
  data?: TripPlan | any
}

/** SSE 事件联合 data 类型 */
export type SSEEventData =
  | SSEAgentData
  | SSEToolData
  | SSEProgressData
  | SSEDoneData
  | SSEErrorData
