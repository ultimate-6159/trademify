<template>
  <div class="space-y-6 pb-20 md:pb-0">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
      <div>
        <h1 class="text-2xl sm:text-3xl font-bold text-white flex items-center gap-2">
          <span class="text-orange-500">&#x1F525;</span> Real-Time Trading
        </h1>
        <p class="text-gray-400 text-sm mt-1">
          Live signal analysis from backend • {{ connectionStatus }}
        </p>
      </div>
      
      <div class="flex items-center gap-3">
        <!-- Symbol Selector -->
        <select
          v-model="selectedSymbol"
          @change="refreshAll"
          class="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500"
        >
          <option v-for="symbol in symbols" :key="symbol" :value="symbol">
            {{ symbol }}
          </option>
        </select>
        
        <!-- Auto Refresh Toggle -->
        <button 
          @click="toggleAutoRefresh"
          :class="autoRefresh ? 'bg-green-600' : 'bg-gray-600'"
          class="px-3 py-2 rounded-lg text-white text-sm flex items-center gap-1"
        >
          <span :class="{ 'animate-spin': autoRefresh && isLoading }">??</span>
          {{ autoRefresh ? 'Auto' : 'Manual' }}
        </button>
      </div>
    </div>

    <!-- Connection Status -->
    <div 
      v-if="!isConnected"
      class="bg-red-900/30 border border-red-500 rounded-lg p-4"
    >
      <div class="flex items-center gap-3">
        <span class="text-2xl">??</span>
        <div>
          <div class="text-red-400 font-semibold">Backend Not Connected</div>
          <div class="text-gray-400 text-sm">
            Make sure the API server is running on port 8000
          </div>
        </div>
      </div>
    </div>

    <!-- Bot Status Banner -->
    <div 
      class="rounded-lg p-4"
      :class="botRunning ? 'bg-green-900/30 border border-green-500' : 'bg-yellow-900/30 border border-yellow-500'"
    >
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span class="text-2xl">{{ botRunning ? '??' : '??' }}</span>
          <div>
            <div :class="botRunning ? 'text-green-400' : 'text-yellow-400'" class="font-semibold">
              {{ botRunning ? 'Bot Running' : 'Bot Stopped' }}
            </div>
            <div class="text-gray-400 text-sm">
              {{ botRunning ? `Analyzing ${symbols.join(', ')}` : 'Start the bot to get signals' }}
            </div>
          </div>
        </div>
        <div class="text-gray-400 text-sm">
          Signal Mode: <span class="text-white font-semibold">{{ signalMode.toUpperCase() }}</span>
        </div>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Technical Signal Panel -->
      <TechnicalSignalPanel 
        :symbol="selectedSymbol"
        @signal-updated="onSignalUpdated"
      />
      
      <!-- 20-Layer Status -->
      <LayerStatusPanel 
        :symbol="selectedSymbol"
      />
    </div>

    <!-- Account & Positions -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Account Info -->
      <div class="bg-gray-800 rounded-lg p-4">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>??</span> Account
        </h3>
        <div class="space-y-2">
          <div class="flex justify-between">
            <span class="text-gray-400">Balance</span>
            <span class="text-white font-semibold">${{ formatNumber(account.balance) }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Equity</span>
            <span class="text-white font-semibold">${{ formatNumber(account.equity) }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Free Margin</span>
            <span class="text-white font-semibold">${{ formatNumber(account.free_margin) }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Margin Level</span>
            <span 
              class="font-semibold"
              :class="account.margin_level > 500 ? 'text-green-400' : account.margin_level > 200 ? 'text-yellow-400' : 'text-red-400'"
            >
              {{ account.margin_level ? `${account.margin_level.toFixed(0)}%` : '---' }}
            </span>
          </div>
        </div>
      </div>

      <!-- Open Positions -->
      <div class="bg-gray-800 rounded-lg p-4">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>??</span> Open Positions ({{ positions.length }})
        </h3>
        <div v-if="positions.length === 0" class="text-gray-500 text-center py-4">
          No open positions
        </div>
        <div v-else class="space-y-2 max-h-40 overflow-y-auto">
          <div 
            v-for="pos in positions" 
            :key="pos.id"
            class="bg-gray-700 rounded p-2 text-sm"
          >
            <div class="flex justify-between">
              <span class="text-white font-semibold">{{ pos.symbol }}</span>
              <span 
                :class="pos.side === 'BUY' ? 'text-green-400' : 'text-red-400'"
              >
                {{ pos.side }}
              </span>
            </div>
            <div class="flex justify-between text-xs">
              <span class="text-gray-400">P&L</span>
              <span :class="pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'">
                {{ pos.pnl >= 0 ? '+' : '' }}${{ pos.pnl?.toFixed(2) || '0.00' }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Daily Stats -->
      <div class="bg-gray-800 rounded-lg p-4">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span>??</span> Today's Performance
        </h3>
        <div class="grid grid-cols-2 gap-3">
          <div class="text-center">
            <div class="text-2xl font-bold text-white">{{ dailyStats.trades || 0 }}</div>
            <div class="text-xs text-gray-400">Trades</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold text-green-400">{{ dailyStats.wins || 0 }}</div>
            <div class="text-xs text-gray-400">Wins</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold text-red-400">{{ dailyStats.losses || 0 }}</div>
            <div class="text-xs text-gray-400">Losses</div>
          </div>
          <div class="text-center">
            <div 
              class="text-2xl font-bold"
              :class="(dailyStats.pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'"
            >
              {{ (dailyStats.pnl || 0) >= 0 ? '+' : '' }}${{ (dailyStats.pnl || 0).toFixed(2) }}
            </div>
            <div class="text-xs text-gray-400">P&L</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Signal History -->
    <div class="bg-gray-800 rounded-lg p-4">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>??</span> Recent Signals
      </h3>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-gray-400 border-b border-gray-700">
              <th class="text-left py-2">Time</th>
              <th class="text-left py-2">Symbol</th>
              <th class="text-left py-2">Signal</th>
              <th class="text-left py-2">Quality</th>
              <th class="text-left py-2">Confidence</th>
              <th class="text-left py-2">Action</th>
            </tr>
          </thead>
          <tbody>
            <tr 
              v-for="signal in signalHistory.slice(0, 10)" 
              :key="signal.id"
              class="border-b border-gray-700"
            >
              <td class="py-2 text-gray-400">{{ formatTime(signal.timestamp) }}</td>
              <td class="py-2 text-white">{{ signal.symbol }}</td>
              <td class="py-2">
                <span :class="signalColorClass(signal.signal)">
                  {{ signalEmoji(signal.signal) }} {{ signal.signal }}
                </span>
              </td>
              <td class="py-2">
                <span :class="qualityColorClass(signal.quality)">
                  {{ signal.quality }}
                </span>
              </td>
              <td class="py-2 text-gray-300">{{ (signal.confidence || 0).toFixed(1) }}%</td>
              <td class="py-2">
                <span 
                  class="px-2 py-0.5 rounded text-xs"
                  :class="signal.action === 'EXECUTED' ? 'bg-green-600' : 'bg-gray-600'"
                >
                  {{ signal.action || 'SKIP' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import TechnicalSignalPanel from '@/components/TechnicalSignalPanel.vue'
import LayerStatusPanel from '@/components/LayerStatusPanel.vue'

// API Base URL
const getApiBase = () => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/api\/v1\/?$/, '')
  }
  const hostname = window.location.hostname
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `http://${hostname}:8000`
  }
  return 'http://localhost:8000'
}
const API_BASE = getApiBase()

// State
const symbols = ref(['EURUSDm', 'GBPUSDm', 'XAUUSDm'])
const selectedSymbol = ref('XAUUSDm')
const isConnected = ref(true)
const isLoading = ref(false)
const autoRefresh = ref(true)
const botRunning = ref(false)
const signalMode = ref('technical')

const account = ref({
  balance: 0,
  equity: 0,
  free_margin: 0,
  margin_level: 0
})

const positions = ref([])
const dailyStats = ref({
  trades: 0,
  wins: 0,
  losses: 0,
  pnl: 0
})

const signalHistory = ref([])

// Polling
let pollInterval = null

// Computed
const connectionStatus = computed(() => {
  return isConnected.value ? '?? Connected' : '?? Disconnected'
})

// Methods
function formatNumber(num) {
  if (!num) return '0.00'
  return num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function formatTime(ts) {
  if (!ts) return '---'
  try {
    const date = new Date(ts)
    return date.toLocaleTimeString()
  } catch {
    return ts
  }
}

function signalEmoji(signal) {
  const map = {
    'STRONG_BUY': '??',
    'BUY': '??',
    'WAIT': '??',
    'SELL': '??',
    'STRONG_SELL': '??'
  }
  return map[signal] || '?'
}

function signalColorClass(signal) {
  if (signal === 'STRONG_BUY' || signal === 'BUY') return 'text-green-400'
  if (signal === 'STRONG_SELL' || signal === 'SELL') return 'text-red-400'
  return 'text-gray-400'
}

function qualityColorClass(quality) {
  if (quality === 'PREMIUM') return 'text-purple-400'
  if (quality === 'HIGH') return 'text-green-400'
  if (quality === 'MEDIUM') return 'text-yellow-400'
  if (quality === 'LOW') return 'text-orange-400'
  return 'text-gray-400'
}

function toggleAutoRefresh() {
  autoRefresh.value = !autoRefresh.value
  if (autoRefresh.value) {
    startPolling()
  } else {
    stopPolling()
  }
}

function onSignalUpdated(signal) {
  // Add to history
  if (signal && signal.symbol) {
    const historyItem = {
      id: `${signal.symbol}_${Date.now()}`,
      symbol: signal.symbol,
      signal: signal.signal,
      quality: signal.quality,
      confidence: signal.enhanced_confidence,
      timestamp: signal.timestamp || new Date().toISOString(),
      action: signal.signal === 'WAIT' ? 'SKIP' : 'PENDING'
    }
    signalHistory.value.unshift(historyItem)
    if (signalHistory.value.length > 50) {
      signalHistory.value = signalHistory.value.slice(0, 50)
    }
  }
}

async function fetchBotStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/bot/status`)
    if (response.ok) {
      const data = await response.json()
      isConnected.value = true
      botRunning.value = data.running || false
      signalMode.value = data.signal_mode || data.config?.signal_mode || 'technical'
      
      if (data.symbols && data.symbols.length > 0) {
        symbols.value = data.symbols
      }
      
      if (data.daily_stats) {
        dailyStats.value = data.daily_stats
      }
    } else {
      isConnected.value = false
    }
  } catch (error) {
    console.error('Failed to fetch bot status:', error)
    isConnected.value = false
  }
}

async function fetchAccount() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/trading/account`)
    if (response.ok) {
      const data = await response.json()
      account.value = {
        balance: data.balance || 0,
        equity: data.equity || 0,
        free_margin: data.free_margin || 0,
        margin_level: data.margin_level || 0
      }
    }
  } catch (error) {
    console.error('Failed to fetch account:', error)
  }
}

async function fetchPositions() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/trading/positions`)
    if (response.ok) {
      const data = await response.json()
      positions.value = data.positions || []
    }
  } catch (error) {
    console.error('Failed to fetch positions:', error)
  }
}

async function refreshAll() {
  isLoading.value = true
  try {
    await Promise.all([
      fetchBotStatus(),
      fetchAccount(),
      fetchPositions()
    ])
  } finally {
    isLoading.value = false
  }
}

function startPolling() {
  if (pollInterval) return
  pollInterval = setInterval(refreshAll, 10000) // Poll every 10 seconds
}

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
  }
}

// Lifecycle
onMounted(() => {
  refreshAll()
  if (autoRefresh.value) {
    startPolling()
  }
})

onUnmounted(() => {
  stopPolling()
})
</script>
