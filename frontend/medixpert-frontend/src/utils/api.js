/**
 * API utility functions for MediXpert
 * Handles communication with Flask backend
 */

const API_BASE_URL = '/api'

// Generic API request function
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`
  
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  }
  
  const config = { ...defaultOptions, ...options }
  
  try {
    const response = await fetch(url, config)
    const data = await response.json()
    
    if (!response.ok) {
      throw new Error(data.error || `HTTP error! status: ${response.status}`)
    }
    
    return data
  } catch (error) {
    console.error(`API request failed for ${endpoint}:`, error)
    throw error
  }
}

// Disease prediction API
export async function predictDisease(symptoms) {
  return apiRequest('/predict-disease', {
    method: 'POST',
    body: JSON.stringify({ symptoms })
  })
}

// Chat API
export async function sendChatMessage(message) {
  return apiRequest('/chat', {
    method: 'POST',
    body: JSON.stringify({ message })
  })
}

// Doctors API
export async function getDoctors(filters = {}) {
  const params = new URLSearchParams()
  
  if (filters.specialty) {
    params.append('specialty', filters.specialty)
  }
  
  if (filters.search) {
    params.append('search', filters.search)
  }
  
  const endpoint = params.toString() ? `/doctors?${params.toString()}` : '/doctors'
  return apiRequest(endpoint)
}

export async function getDoctor(doctorId) {
  return apiRequest(`/doctors/${doctorId}`)
}

// Appointments API
export async function getAppointments(status = '') {
  const endpoint = status ? `/appointments?status=${status}` : '/appointments'
  return apiRequest(endpoint)
}

export async function bookAppointment(appointmentData) {
  return apiRequest('/appointments', {
    method: 'POST',
    body: JSON.stringify(appointmentData)
  })
}

// Reports API
export async function getReports() {
  return apiRequest('/reports')
}

export async function uploadReport(reportData) {
  return apiRequest('/reports', {
    method: 'POST',
    body: JSON.stringify(reportData)
  })
}

// Dashboard API
export async function getDashboardStats() {
  return apiRequest('/dashboard/stats')
}

// Health check API
export async function healthCheck() {
  return apiRequest('/health')
}

