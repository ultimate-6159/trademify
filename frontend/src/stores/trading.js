/**
 * Trading Store
 * State management สำหรับระบบเทรดอัตโนมัติ
 * รองรับ Firebase real-time sync สำหรับหลาย browser/VPS
 * 
 * Auto-Sync: ดึงข้อมูลจาก Backend ทุก 10 วินาที
 * เพื่อให้ทุกอุปกรณ์เห็นข้อมูลตรงกัน (fallback เมื่อไม่มี Firebase)
 */
import { defineStore } from 'pinia'
import { ref, computed, onUnmounted } from 'vue'
import { 
  initFirebase, 
  isFirebaseAvailable,
  subscribeToPositions,
  subscribeToDailyStats,
  subscribeToNodes,
  subscribeToTradeHistory,
  subscribeToSettings,
  unsubscribeAll
} from '../services/firebase'

// Polling interval (ms) - ดึงข้อมูลทุก 10 วินาที (fallback when no Firebase)
const SYNC_INTERVAL = 10000

// Auto-detect API URL based on hostname (for VPS deployment)
function getApiBase() {
  const hostname = window.location.hostname
  // If running on VPS (not localhost), use same hostname for API
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `http://${hostname}:8000`
  }
  // Development: use VITE_API_URL or default to localhost
  const rawApiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  return rawApiUrl.replace(/\/api\/v1\/?$/, '')
}
const API_BASE = getApiBase()

export const useTradingStore = defineStore('trading', () => {
  // State
  const enabled = ref(false)
  const status = ref('stopped') // running, paused, stopped, disconnected
  const paperTrading = ref(true)
  const positions = ref([])
  const stats = ref({
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    total_pnl: 0,
  })
  const settings = ref({
    max_risk_per_trade: 2.0,
    max_daily_loss: 5.0,
    max_positions: 5,
    min_confidence: 70,
    allowed_signals: ['STRONG_BUY', 'STRONG_SELL'],
  })
  
  // Connection & Error State
  const apiConnected = ref(false)
  const firebaseConnected = ref(false)
  const lastError = ref(null)
  const errorMessage = ref('')
  
  // Cluster State (multi-VPS)
  const clusterNodes = ref([])
  const tradeHistory = ref([])
  
  // Firebase unsubscribe functions
  const unsubscribers = []
  
  // Auto-Sync State (fallback when no Firebase)
  let syncIntervalId = null
  const lastSyncTime = ref(null)
  const isSyncing = ref(false)

  // Computed
  const openPositions = computed(() => positions.value.length)
  
  const todayPnl = computed(() => 
    positions.value.reduce((sum, p) => sum + (p.pnl || 0), 0)
  )
  
  const winRate = computed(() => {
    const total = stats.value.total_trades
    if (total === 0) return 0
    return (stats.value.winning_trades / total) * 100
  })
  
  const riskUsed = computed(() => {
    // Calculate based on daily loss
    return Math.abs(Math.min(0, stats.value.total_pnl) / 100) // Simplified calculation
  })
  
  const maxDailyLoss = computed(() => settings.value.max_daily_loss)
  
  // Connection status message
  const statusMessage = computed(() => {
    if (!apiConnected.value) {
      return '❌ API Disconnected - Backend server not running'
    }
    if (firebaseConnected.value) {
      return status.value === 'running' 
        ? '✅ Trading Active (Real-time Sync)' 
        : status.value === 'paused'
        ? '⏸️ Trading Paused (Real-time Sync)'
        : '⏹️ Trading Stopped (Real-time Sync)'
    }
    if (status.value === 'running') {
      return '✅ Trading Active'
    }
    if (status.value === 'paused') {
      return '⏸️ Trading Paused'
    }
    return '⏹️ Trading Stopped'
  })
  
  // Total cluster positions
  const totalClusterPositions = computed(() => positions.value.length)
  
  // Active nodes count
  const activeNodesCount = computed(() => clusterNodes.value.length)

  // =====================
  // Firebase Real-time Setup
  // =====================
  
  function initializeFirebaseSync() {
    // Try to initialize Firebase
    const initialized = initFirebase()
    firebaseConnected.value = initialized
    
    if (!initialized) {
      console.log('[Trading Store] Firebase not available, using API polling')
      return
    }
    
    console.log('[Trading Store] Setting up Firebase real-time listeners')
    
    // Subscribe to positions
    const unsubPositions = subscribeToPositions((newPositions) => {
      console.log('[Firebase] Positions updated:', newPositions.length)
      positions.value = newPositions
    })
    unsubscribers.push(unsubPositions)
    
    // Subscribe to daily stats
    const unsubStats = subscribeToDailyStats((newStats) => {
      console.log('[Firebase] Daily stats updated:', newStats)
      stats.value = {
        ...stats.value,
        total_trades: newStats.total_trades || 0,
        winning_trades: newStats.winning_trades || 0,
        losing_trades: newStats.losing_trades || 0,
        total_pnl: newStats.total_pnl || 0,
      }
    })
    unsubscribers.push(unsubStats)
    
    // Subscribe to cluster nodes
    const unsubNodes = subscribeToNodes((nodes) => {
      console.log('[Firebase] Cluster nodes:', nodes.length)
      clusterNodes.value = nodes
    })
    unsubscribers.push(unsubNodes)
    
    // Subscribe to trade history
    const unsubHistory = subscribeToTradeHistory((trades) => {
      console.log('[Firebase] Trade history:', trades.length)
      tradeHistory.value = trades
    })
    unsubscribers.push(unsubHistory)
    
    // Subscribe to settings
    const unsubSettings = subscribeToSettings((newSettings) => {
      console.log('[Firebase] Settings updated')
      if (newSettings.risk) {
        settings.value = {
          ...settings.value,
          max_risk_per_trade: newSettings.risk.max_risk_per_trade || settings.value.max_risk_per_trade,
          max_daily_loss: newSettings.risk.max_daily_loss || settings.value.max_daily_loss,
          max_positions: newSettings.risk.max_positions || settings.value.max_positions,
        }
      }
    })
    unsubscribers.push(unsubSettings)
  }
  
  function cleanupFirebase() {
    unsubscribers.forEach(unsub => {
      if (typeof unsub === 'function') unsub()
    })
    unsubscribers.length = 0
    unsubscribeAll()
  }

  // Actions
  async function fetchStatus() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/status`)
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      
      const data = await response.json()
      
      apiConnected.value = true
      enabled.value = data.enabled || false
      status.value = data.running ? 'running' : (data.enabled ? 'paused' : 'stopped')
      lastError.value = null
      errorMessage.value = ''
      
      if (data.stats) {
        stats.value = data.stats
      }
      
      return data
    } catch (error) {
      apiConnected.value = false
      status.value = 'disconnected'
      lastError.value = error
      errorMessage.value = `Cannot connect to trading API: ${error.message}`
      console.warn('[Trading] API disconnected:', error.message)
      return null
    }
  }

  async function fetchSettings() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/settings`)
      const data = await response.json()
      
      paperTrading.value = data.paper_trading
      settings.value = {
        max_risk_per_trade: data.risk?.max_risk_per_trade || 2.0,
        max_daily_loss: data.risk?.max_daily_loss || 5.0,
        max_positions: data.risk?.max_positions || 5,
        min_confidence: data.signals?.min_confidence || 70,
        allowed_signals: data.signals?.allowed_signals || ['STRONG_BUY', 'STRONG_SELL'],
      }
      
      return data
    } catch (error) {
      console.error('Failed to fetch settings:', error)
    }
  }

  // =====================
  // Auto-Sync Functions (Fallback when no Firebase)
  // =====================
  
  /**
   * Sync ข้อมูลทั้งหมดจาก Backend
   * เรียกทุก SYNC_INTERVAL เพื่อให้ทุกอุปกรณ์เห็นข้อมูลตรงกัน
   */
  async function syncFromBackend() {
    if (isSyncing.value) return // ป้องกัน concurrent sync
    
    isSyncing.value = true
    try {
      // ดึงข้อมูลพร้อมกันเพื่อความเร็ว
      await Promise.all([
        fetchStatus(),
        fetchSettings(),
        fetchPositions(),
      ])
      lastSyncTime.value = new Date()
      console.debug('[Trading] Synced from backend at', lastSyncTime.value.toLocaleTimeString())
    } catch (error) {
      console.warn('[Trading] Sync failed:', error.message)
    } finally {
      isSyncing.value = false
    }
  }
  
  /**
   * เริ่ม Auto-Sync polling
   * จะดึงข้อมูลจาก Backend ทุก SYNC_INTERVAL (10 วินาที)
   * ใช้เป็น fallback เมื่อไม่มี Firebase
   */
  function startAutoSync() {
    // ถ้ามี Firebase ไม่ต้องใช้ polling
    if (firebaseConnected.value) {
      console.log('[Trading] Firebase connected, skipping auto-sync polling')
      return
    }
    
    if (syncIntervalId) {
      console.debug('[Trading] Auto-sync already running')
      return
    }
    
    // Sync ทันทีครั้งแรก
    syncFromBackend()
    
    // ตั้ง interval สำหรับ sync ต่อไป
    syncIntervalId = setInterval(syncFromBackend, SYNC_INTERVAL)
    console.log(`[Trading] Auto-sync started (every ${SYNC_INTERVAL/1000}s)`)
  }
  
  /**
   * หยุด Auto-Sync polling
   */
  function stopAutoSync() {
    if (syncIntervalId) {
      clearInterval(syncIntervalId)
      syncIntervalId = null
      console.log('[Trading] Auto-sync stopped')
    }
  }
  
  /**
   * Force sync ทันที (สำหรับกรณีผู้ใช้ต้องการ refresh)
   */
  async function forceSync() {
    console.log('[Trading] Force sync triggered')
    await syncFromBackend()
  }

  async function updateSettings(newSettings) {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSettings),
      })
      
      if (response.ok) {
        await fetchSettings()
        return true
      }
    } catch (error) {
      console.error('Failed to update settings:', error)
    }
    return false
  }

  async function startTrading() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/start`, {
        method: 'POST',
      })
      
      if (response.ok) {
        enabled.value = true
        status.value = 'running'
        return true
      }
    } catch (error) {
      console.error('Failed to start trading:', error)
    }
    return false
  }

  async function stopTrading() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/stop`, {
        method: 'POST',
      })
      
      if (response.ok) {
        enabled.value = false
        status.value = 'stopped'
        return true
      }
    } catch (error) {
      console.error('Failed to stop trading:', error)
    }
    return false
  }

  async function pauseTrading() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/pause`, {
        method: 'POST',
      })
      
      if (response.ok) {
        status.value = 'paused'
        return true
      }
    } catch (error) {
      console.error('Failed to pause trading:', error)
    }
    return false
  }

  async function resumeTrading() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/resume`, {
        method: 'POST',
      })
      
      if (response.ok) {
        status.value = 'running'
        return true
      }
    } catch (error) {
      console.error('Failed to resume trading:', error)
    }
    return false
  }

  async function fetchPositions() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/positions`)
      const data = await response.json()
      
      positions.value = data.positions || []
      return data
    } catch (error) {
      console.error('Failed to fetch positions:', error)
    }
  }

  async function openPosition(orderData) {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/positions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(orderData),
      })
      
      if (response.ok) {
        await fetchPositions()
        return await response.json()
      }
    } catch (error) {
      console.error('Failed to open position:', error)
    }
    return null
  }

  async function closePosition(positionId) {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/positions/${positionId}`, {
        method: 'DELETE',
      })
      
      if (response.ok) {
        await fetchPositions()
        return await response.json()
      }
    } catch (error) {
      console.error('Failed to close position:', error)
    }
    return null
  }

  async function closeAllPositions() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/positions`, {
        method: 'DELETE',
      })
      
      if (response.ok) {
        await fetchPositions()
        return await response.json()
      }
    } catch (error) {
      console.error('Failed to close all positions:', error)
    }
    return null
  }

  async function modifyPosition(positionId, modifications) {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/positions/${positionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(modifications),
      })
      
      if (response.ok) {
        await fetchPositions()
        return await response.json()
      }
    } catch (error) {
      console.error('Failed to modify position:', error)
    }
    return null
  }

  async function fetchHistory(limit = 50) {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/history?limit=${limit}`)
      return await response.json()
    } catch (error) {
      console.error('Failed to fetch history:', error)
    }
    return null
  }

  async function fetchStats() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/stats`)
      const data = await response.json()
      
      if (data.engine) {
        stats.value = data.engine
      }
      
      return data
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  async function getPrice(symbol) {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/price/${encodeURIComponent(symbol)}`)
      
      if (!response.ok) {
        // Don't log 404 as error - it's expected when API is not fully set up
        if (response.status !== 404) {
          console.warn(`[Trading] Price fetch failed: HTTP ${response.status}`)
        }
        return { price: null, error: `HTTP ${response.status}`, source: 'error' }
      }
      
      const data = await response.json()
      return data
    } catch (error) {
      // Network error - API likely not running
      console.warn('[Trading] Price API unavailable:', error.message)
      return { 
        price: null, 
        error: 'API unavailable', 
        source: 'error',
        message: 'Cannot fetch price - Backend API not running'
      }
    }
  }

  async function sendSignal(signalData) {
    try {
      const response = await fetch(`${API_BASE}/api/v1/trading/signal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(signalData),
      })
      
      if (response.ok) {
        await fetchPositions()
        return await response.json()
      }
    } catch (error) {
      console.error('Failed to send signal:', error)
    }
    return null
  }

  // Market Status
  const marketStatus = ref({
    status: 'UNKNOWN',
    is_tradeable: false,
    message: 'Loading...',
    mt5_connected: false
  })

  async function fetchMarketStatus(symbol = 'EURUSD') {
    try {
      const response = await fetch(`${API_BASE}/api/v1/market/status?symbol=${symbol}`)
      
      if (response.ok) {
        const data = await response.json()
        marketStatus.value = data
        return data
      }
    } catch (error) {
      console.warn('[Trading] Market status unavailable:', error.message)
    }
    return null
  }

  // MT5 Account
  async function fetchMT5Account() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/mt5/account`)
      if (response.ok) {
        return await response.json()
      }
    } catch (error) {
      console.warn('[Trading] MT5 account unavailable:', error.message)
    }
    return null
  }

  // Reconnect MT5
  async function reconnectMT5() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/mt5/reconnect`, {
        method: 'POST'
      })
      if (response.ok) {
        const data = await response.json()
        await fetchMarketStatus()
        return data
      }
    } catch (error) {
      console.error('Failed to reconnect MT5:', error)
    }
    return null
  }

  return {
    // State
    enabled,
    status,
    paperTrading,
    positions,
    stats,
    settings,
    apiConnected,
    firebaseConnected,
    lastError,
    errorMessage,
    marketStatus,
    clusterNodes,
    tradeHistory,
    lastSyncTime,
    isSyncing,
    
    // Computed
    openPositions,
    todayPnl,
    winRate,
    riskUsed,
    maxDailyLoss,
    statusMessage,
    totalClusterPositions,
    activeNodesCount,
    
    // Actions
    initializeFirebaseSync,
    cleanupFirebase,
    fetchStatus,
    fetchSettings,
    updateSettings,
    startTrading,
    stopTrading,
    pauseTrading,
    resumeTrading,
    fetchPositions,
    openPosition,
    closePosition,
    closeAllPositions,
    modifyPosition,
    fetchHistory,
    fetchStats,
    getPrice,
    sendSignal,
    fetchMarketStatus,
    fetchMT5Account,
    reconnectMT5,
    
    // Auto-Sync (fallback when no Firebase)
    startAutoSync,
    stopAutoSync,
    forceSync,
    syncFromBackend,
  }
})
