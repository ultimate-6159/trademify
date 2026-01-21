<template>
  <div 
    :class="bannerClass"
    class="rounded-lg p-4 mb-4 transition-all duration-300"
  >
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <!-- Status Icon -->
        <div :class="iconClass" class="text-3xl animate-pulse">
          {{ statusIcon }}
        </div>
        
        <div>
          <h3 :class="titleClass" class="font-bold text-lg">
            {{ statusTitle }}
          </h3>
          <p class="text-sm opacity-80">{{ statusMessage }}</p>
        </div>
      </div>
      
      <div class="text-right">
        <!-- Time until open (if closed) -->
        <div v-if="!marketInfo.is_tradeable && marketInfo.time_until_open" class="text-right">
          <span class="text-xs opacity-60">Opens in</span>
          <p class="text-xl font-bold">{{ marketInfo.time_until_open }}</p>
        </div>
        
        <!-- Connection status + Reconnect button -->
        <div class="flex items-center gap-2 mt-1">
          <span 
            :class="mt5Connected ? 'bg-green-500' : 'bg-red-500'"
            class="w-2 h-2 rounded-full"
          ></span>
          <span class="text-xs" :class="mt5Connected ? 'text-green-400' : 'text-red-400'">
            {{ mt5Connected ? '🟢 MT5 Live' : '🔴 MT5 Disconnected' }}
          </span>
          <button 
            v-if="!mt5Connected"
            @click="reconnectMT5"
            :disabled="reconnecting"
            class="ml-2 px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded"
          >
            {{ reconnecting ? '...' : '🔄 Reconnect' }}
          </button>
        </div>
      </div>
    </div>
    
    <!-- MT5 Not Connected Warning -->
    <div v-if="!mt5Connected && !loading" class="mt-3 pt-3 border-t border-red-500/30 bg-red-900/20 rounded p-3">
      <p class="text-red-400 text-sm font-semibold mb-2">
        ❌ MT5 Terminal ไม่ได้เชื่อมต่อ - ไม่สามารถดึงราคาได้
      </p>
      <ul class="text-xs text-red-300 space-y-1">
        <li>1. ตรวจสอบว่า MT5 Terminal เปิดอยู่บน Windows VPS</li>
        <li>2. ตรวจสอบ .env มีค่า MT5_LOGIN, MT5_PASSWORD, MT5_SERVER ถูกต้อง</li>
        <li>3. กดปุ่ม Reconnect หรือ restart backend</li>
      </ul>
    </div>
    
    <!-- Trading Warning -->
    <div v-else-if="!marketInfo.is_tradeable" class="mt-3 pt-3 border-t border-white/10">
      <p class="text-sm">
        ⚠️ Trading disabled - Market is {{ marketInfo.status }}
      </p>
    </div>
    
    <!-- Session Info (when open) -->
    <div v-if="marketInfo.is_tradeable && session && mt5Connected" class="mt-2">
      <span class="text-xs bg-white/10 px-2 py-1 rounded">
        📍 Active Session: {{ session }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

// Auto-detect API URL
function getApiBase() {
  const hostname = window.location.hostname
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `http://${hostname}:8000`
  }
  return import.meta.env.VITE_API_URL?.replace(/\/api\/v1\/?$/, '') || 'http://localhost:8000'
}
const API_BASE = getApiBase()

const props = defineProps({
  symbol: {
    type: String,
    default: 'EURUSD'
  }
})

const marketInfo = ref({
  status: 'UNKNOWN',
  is_tradeable: false,
  message: 'Loading...',
  message_th: 'กำลังโหลด...',
  time_until_open: null
})

const mt5Connected = ref(false)
const session = ref(null)
const loading = ref(true)
const reconnecting = ref(false)

// Reconnect MT5
async function reconnectMT5() {
  reconnecting.value = true
  try {
    const response = await fetch(`${API_BASE}/api/v1/mt5/reconnect`, { method: 'POST' })
    const data = await response.json()
    if (data.status === 'connected') {
      mt5Connected.value = true
      alert('✅ MT5 reconnected successfully!')
      await fetchMarketStatus()
    } else {
      alert('❌ Could not connect to MT5: ' + (data.message || 'Unknown error'))
    }
  } catch (error) {
    alert('❌ Reconnect failed: ' + error.message)
  } finally {
    reconnecting.value = false
  }
}

// Computed classes
const bannerClass = computed(() => {
  const base = 'border'
  if (loading.value) return `${base} bg-gray-800 border-gray-600`
  
  // MT5 not connected = red banner
  if (!mt5Connected.value) {
    return `${base} bg-red-900/40 border-red-500/50`
  }
  
  switch (marketInfo.value.status) {
    case 'OPEN':
      return `${base} bg-green-900/40 border-green-500/50`
    case 'CLOSED':
    case 'WEEKEND':
      return `${base} bg-yellow-900/40 border-yellow-500/50`
    case 'PRE_MARKET':
    case 'POST_MARKET':
      return `${base} bg-yellow-900/40 border-yellow-500/50`
    default:
      return `${base} bg-gray-800 border-gray-600`
  }
})

const iconClass = computed(() => {
  if (loading.value) return 'text-gray-400'
  if (!mt5Connected.value) return 'text-red-400'
  
  switch (marketInfo.value.status) {
    case 'OPEN': return 'text-green-400'
    case 'CLOSED':
    case 'WEEKEND': return 'text-yellow-400'
    default: return 'text-yellow-400'
  }
})

const titleClass = computed(() => {
  if (loading.value) return 'text-gray-300'
  if (!mt5Connected.value) return 'text-red-400'
  
  switch (marketInfo.value.status) {
    case 'OPEN': return 'text-green-400'
    case 'CLOSED':
    case 'WEEKEND': return 'text-yellow-400'
    default: return 'text-yellow-400'
  }
})

const statusIcon = computed(() => {
  if (loading.value) return '⏳'
  if (!mt5Connected.value) return '❌'
  
  switch (marketInfo.value.status) {
    case 'OPEN': return '🟢'
    case 'CLOSED':
    case 'WEEKEND': return '🟡'
    case 'PRE_MARKET': return '🌅'
    case 'POST_MARKET': return '🌙'
    default: return '❓'
  }
})

const statusTitle = computed(() => {
  if (loading.value) return 'Checking Market...'
  if (!mt5Connected.value) return 'MT5 Not Connected'
  
  switch (marketInfo.value.status) {
    case 'OPEN': return 'Market OPEN'
    case 'CLOSED': return 'Market CLOSED'
    case 'WEEKEND': return 'Weekend - Market Closed'
    case 'PRE_MARKET': return 'Pre-Market'
    case 'POST_MARKET': return 'After Hours'
    default: return 'Market Status Unknown'
  }
})

const statusMessage = computed(() => {
  if (loading.value) return 'Connecting to market...'
  if (!mt5Connected.value) return 'กรุณาเปิด MT5 Terminal และกด Reconnect'
  return marketInfo.value.message_th || marketInfo.value.message
})

async function fetchMarketStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/mt5/market-status?symbol=${props.symbol}`)
    if (response.ok) {
      const data = await response.json()
      marketInfo.value = data
      mt5Connected.value = data.mt5_connected || false
      
      // Extract session from message
      const sessionMatch = data.message?.match(/Session: (\w+)/)
      if (sessionMatch) {
        session.value = sessionMatch[1]
      }
    } else {
      console.warn('Market status API returned:', response.status)
      mt5Connected.value = false
    }
    loading.value = false
  } catch (error) {
    console.error('Failed to fetch market status:', error)
    mt5Connected.value = false
    loading.value = false
  }
}

let interval = null

onMounted(() => {
  fetchMarketStatus()
  // Update every 30 seconds
  interval = setInterval(fetchMarketStatus, 30000)
})

onUnmounted(() => {
  if (interval) clearInterval(interval)
})
</script>

<style scoped>
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}
.animate-pulse {
  animation: pulse 2s ease-in-out infinite;
}
</style>
