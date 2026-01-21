/**
 * Firebase Real-time Service
 * Sync ข้อมูลระหว่าง Browser หลายตัวผ่าน Firebase Realtime Database
 */

import { initializeApp } from 'firebase/app'
import { getDatabase, ref, onValue, off, set, get, push, serverTimestamp } from 'firebase/database'

// Firebase configuration - ต้องตั้งค่าใน .env
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  databaseURL: import.meta.env.VITE_FIREBASE_DATABASE_URL,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID
}

let app = null
let database = null
let isInitialized = false

// Callbacks for different events
const callbacks = {
  positions: [],
  signals: [],
  stats: [],
  nodes: [],
  settings: []
}

// Active listeners
const activeListeners = new Map()

/**
 * Initialize Firebase
 */
export function initFirebase() {
  if (isInitialized) return true
  
  // Check if Firebase config is available
  if (!firebaseConfig.databaseURL) {
    console.warn('[Firebase] Not configured - using API polling mode')
    return false
  }
  
  try {
    app = initializeApp(firebaseConfig)
    database = getDatabase(app)
    isInitialized = true
    console.log('[Firebase] Initialized successfully')
    return true
  } catch (error) {
    console.error('[Firebase] Failed to initialize:', error)
    return false
  }
}

/**
 * Check if Firebase is available
 */
export function isFirebaseAvailable() {
  return isInitialized && database !== null
}

// ===========================================
// POSITIONS - Real-time sync
// ===========================================

/**
 * Listen to positions changes
 * @param {Function} callback - Called when positions change
 */
export function subscribeToPositions(callback) {
  if (!isFirebaseAvailable()) {
    console.warn('[Firebase] Not available, positions will use API polling')
    return () => {}
  }
  
  const positionsRef = ref(database, 'trading/positions')
  
  const unsubscribe = onValue(positionsRef, (snapshot) => {
    const data = snapshot.val()
    const positions = data ? Object.values(data).filter(p => p.status === 'OPEN') : []
    callback(positions)
  }, (error) => {
    console.error('[Firebase] Positions listener error:', error)
  })
  
  callbacks.positions.push(callback)
  activeListeners.set('positions', positionsRef)
  
  return () => {
    off(positionsRef)
    const index = callbacks.positions.indexOf(callback)
    if (index > -1) callbacks.positions.splice(index, 1)
    activeListeners.delete('positions')
  }
}

/**
 * Get all positions once
 */
export async function getPositions() {
  if (!isFirebaseAvailable()) return []
  
  const positionsRef = ref(database, 'trading/positions')
  const snapshot = await get(positionsRef)
  const data = snapshot.val()
  return data ? Object.values(data).filter(p => p.status === 'OPEN') : []
}

// ===========================================
// SIGNALS - Real-time sync
// ===========================================

/**
 * Listen to current signals
 * @param {string} symbol
 * @param {string} timeframe
 * @param {Function} callback
 */
export function subscribeToSignal(symbol, timeframe, callback) {
  if (!isFirebaseAvailable()) {
    console.warn('[Firebase] Not available, signals will use API polling')
    return () => {}
  }
  
  const signalRef = ref(database, `current_signals/${symbol}/${timeframe}`)
  
  const unsubscribe = onValue(signalRef, (snapshot) => {
    const data = snapshot.val()
    if (data) {
      callback(data)
    }
  }, (error) => {
    console.error('[Firebase] Signal listener error:', error)
  })
  
  const key = `signal_${symbol}_${timeframe}`
  callbacks.signals.push({ key, callback })
  activeListeners.set(key, signalRef)
  
  return () => {
    off(signalRef)
    activeListeners.delete(key)
  }
}

// ===========================================
// CLUSTER STATS - Real-time sync
// ===========================================

/**
 * Listen to daily stats
 * @param {Function} callback
 */
export function subscribeToDailyStats(callback) {
  if (!isFirebaseAvailable()) return () => {}
  
  const today = new Date().toISOString().split('T')[0]
  const statsRef = ref(database, `trading/stats/daily/${today}`)
  
  onValue(statsRef, (snapshot) => {
    const data = snapshot.val()
    if (data) {
      callback(data)
    }
  })
  
  activeListeners.set('daily_stats', statsRef)
  
  return () => {
    off(statsRef)
    activeListeners.delete('daily_stats')
  }
}

/**
 * Listen to active nodes
 * @param {Function} callback
 */
export function subscribeToNodes(callback) {
  if (!isFirebaseAvailable()) return () => {}
  
  const nodesRef = ref(database, 'cluster/nodes')
  
  onValue(nodesRef, (snapshot) => {
    const data = snapshot.val()
    const nodes = data ? Object.values(data) : []
    callback(nodes)
  })
  
  activeListeners.set('nodes', nodesRef)
  
  return () => {
    off(nodesRef)
    activeListeners.delete('nodes')
  }
}

// ===========================================
// TRADE HISTORY - Real-time sync
// ===========================================

/**
 * Listen to trade history
 * @param {Function} callback
 * @param {number} limit
 */
export function subscribeToTradeHistory(callback, limit = 50) {
  if (!isFirebaseAvailable()) return () => {}
  
  const historyRef = ref(database, 'trading/history')
  
  onValue(historyRef, (snapshot) => {
    const data = snapshot.val()
    if (data) {
      const trades = Object.values(data)
        .sort((a, b) => new Date(b.closed_at) - new Date(a.closed_at))
        .slice(0, limit)
      callback(trades)
    }
  })
  
  activeListeners.set('history', historyRef)
  
  return () => {
    off(historyRef)
    activeListeners.delete('history')
  }
}

// ===========================================
// SETTINGS - Shared settings
// ===========================================

/**
 * Listen to shared settings
 * @param {Function} callback
 */
export function subscribeToSettings(callback) {
  if (!isFirebaseAvailable()) return () => {}
  
  const settingsRef = ref(database, 'trading/settings')
  
  onValue(settingsRef, (snapshot) => {
    const data = snapshot.val()
    if (data) {
      callback(data)
    }
  })
  
  activeListeners.set('settings', settingsRef)
  
  return () => {
    off(settingsRef)
    activeListeners.delete('settings')
  }
}

/**
 * Update shared settings
 * @param {Object} settings
 */
export async function updateSharedSettings(settings) {
  if (!isFirebaseAvailable()) return false
  
  try {
    const settingsRef = ref(database, 'trading/settings')
    await set(settingsRef, {
      ...settings,
      updated_at: new Date().toISOString()
    })
    return true
  } catch (error) {
    console.error('[Firebase] Failed to update settings:', error)
    return false
  }
}

// ===========================================
// PATTERNS - For visualization
// ===========================================

/**
 * Listen to current pattern
 * @param {string} symbol
 * @param {string} timeframe
 * @param {Function} callback
 */
export function subscribeToPattern(symbol, timeframe, callback) {
  if (!isFirebaseAvailable()) return () => {}
  
  const patternRef = ref(database, `current_patterns/${symbol}/${timeframe}`)
  
  onValue(patternRef, (snapshot) => {
    const data = snapshot.val()
    if (data) {
      callback(data)
    }
  })
  
  const key = `pattern_${symbol}_${timeframe}`
  activeListeners.set(key, patternRef)
  
  return () => {
    off(patternRef)
    activeListeners.delete(key)
  }
}

// ===========================================
// CLEANUP
// ===========================================

/**
 * Unsubscribe from all listeners
 */
export function unsubscribeAll() {
  for (const [key, refObj] of activeListeners) {
    try {
      off(refObj)
    } catch (e) {
      console.warn(`[Firebase] Failed to unsubscribe from ${key}:`, e)
    }
  }
  activeListeners.clear()
  console.log('[Firebase] All listeners removed')
}

// ===========================================
// EXPORT DEFAULT
// ===========================================

export default {
  initFirebase,
  isFirebaseAvailable,
  subscribeToPositions,
  subscribeToSignal,
  subscribeToDailyStats,
  subscribeToNodes,
  subscribeToTradeHistory,
  subscribeToSettings,
  subscribeToPattern,
  updateSharedSettings,
  getPositions,
  unsubscribeAll
}
