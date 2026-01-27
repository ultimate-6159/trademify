<template>
  <div class="bg-gray-800 rounded-lg p-4 sm:p-6">
    <!-- Header -->
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg sm:text-xl font-bold text-white flex items-center gap-2">
        <span class="text-2xl text-orange-500">&#x1F4CA;</span>
        <span>Technical Signal</span>
        <span 
          class="text-xs px-2 py-1 rounded-full"
          :class="signalMode === 'technical' ? 'bg-orange-500' : 'bg-blue-500'"
        >
          {{ signalMode?.toUpperCase() || 'TECHNICAL' }}
        </span>
      </h2>
      <button 
        @click="refreshSignal" 
        :disabled="isLoading"
        class="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1"
      >
        <span :class="{ 'animate-spin': isLoading }">&#x21BB;</span>
        Refresh
      </button>
    </div>

    <!-- Main Signal Display -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <!-- Signal Direction -->
      <div 
        class="rounded-lg p-4 text-center"
        :class="signalBgClass"
      >
        <div class="text-5xl mb-2">{{ signalEmoji }}</div>
        <div class="text-2xl font-bold text-white">{{ signal?.signal || 'WAIT' }}</div>
        <div class="text-sm text-gray-300 mt-1">
          Confidence: {{ displayConfidence.toFixed(1) }}%
        </div>
      </div>

      <!-- Quality -->
      <div class="bg-gray-700 rounded-lg p-4 text-center">
        <div class="text-gray-400 text-sm mb-2">Quality</div>
        <div 
          class="text-2xl font-bold"
          :class="qualityColorClass"
        >
          {{ signal?.quality || 'SKIP' }}
        </div>
        <div class="text-xs text-gray-400 mt-2">
          {{ qualityDescription }}
        </div>
      </div>

      <!-- Price -->
      <div class="bg-gray-700 rounded-lg p-4 text-center">
        <div class="text-gray-400 text-sm mb-2">Current Price</div>
        <div class="text-2xl font-bold text-white">
          {{ formatPrice(signal?.current_price) }}
        </div>
        <div class="text-xs text-gray-400 mt-2">
          {{ signal?.symbol || selectedSymbol }}
        </div>
      </div>
    </div>

    <!-- Technical Scores (Only in Technical Mode) -->
    <div v-if="signalMode === 'technical' && technicalData" class="mb-6">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>??</span> Technical Scores
      </h3>
      
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <!-- Buy Score -->
        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Buy Score</div>
          <div class="flex items-center gap-2">
            <div class="text-xl font-bold text-green-400">
              {{ technicalData.buy_score || 0 }}/10
            </div>
            <div class="flex-1 h-2 bg-gray-600 rounded-full overflow-hidden">
              <div 
                class="h-full bg-green-500"
                :style="{ width: `${(technicalData.buy_score || 0) * 10}%` }"
              />
            </div>
          </div>
        </div>

        <!-- Sell Score -->
        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Sell Score</div>
          <div class="flex items-center gap-2">
            <div class="text-xl font-bold text-red-400">
              {{ technicalData.sell_score || 0 }}/10
            </div>
            <div class="flex-1 h-2 bg-gray-600 rounded-full overflow-hidden">
              <div 
                class="h-full bg-red-500"
                :style="{ width: `${(technicalData.sell_score || 0) * 10}%` }"
              />
            </div>
          </div>
        </div>

        <!-- Session -->
        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Session</div>
          <div 
            class="text-lg font-bold"
            :class="sessionColorClass"
          >
            {{ technicalData.session || 'N/A' }}
          </div>
        </div>

        <!-- Trend -->
        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Trend</div>
          <div 
            class="text-lg font-bold"
            :class="trendColorClass"
          >
            {{ technicalData.trend || 'RANGE' }}
          </div>
        </div>
      </div>

      <!-- Additional Indicators -->
      <div class="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-3">
        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">RSI</div>
          <div class="text-lg font-bold" :class="rsiColorClass">
            {{ (technicalData.rsi || 0).toFixed(1) }}
          </div>
        </div>

        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">ATR</div>
          <div class="text-lg font-bold text-white">
            {{ (technicalData.atr || 0).toFixed(5) }}
          </div>
        </div>

        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Risk/Reward</div>
          <div class="text-lg font-bold text-purple-400">
            1:{{ (riskReward || 0).toFixed(2) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Risk Management -->
    <div v-if="signal?.risk_management" class="mb-6">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>???</span> Risk Management
      </h3>
      
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Entry</div>
          <div class="text-sm font-bold text-white">
            {{ formatPrice(signal.current_price) }}
          </div>
        </div>

        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Stop Loss</div>
          <div class="text-sm font-bold text-red-400">
            {{ formatPrice(signal.risk_management.stop_loss) }}
          </div>
        </div>

        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Take Profit</div>
          <div class="text-sm font-bold text-green-400">
            {{ formatPrice(signal.risk_management.take_profit) }}
          </div>
        </div>

        <div class="bg-gray-700 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">R:R Ratio</div>
          <div class="text-sm font-bold text-purple-400">
            1:{{ (signal.risk_management.risk_reward || 0).toFixed(2) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Factors -->
    <div v-if="signal?.factors" class="mb-6">
      <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span>??</span> Signal Factors
      </h3>
      
      <div class="space-y-2">
        <!-- Bullish Factors -->
        <div v-if="signal.factors.bullish?.length > 0" class="bg-green-900/30 rounded-lg p-3">
          <div class="text-green-400 text-sm font-semibold mb-2">? Bullish Factors</div>
          <div class="flex flex-wrap gap-2">
            <span 
              v-for="(factor, idx) in signal.factors.bullish" 
              :key="'bullish-'+idx"
              class="bg-green-600/30 text-green-300 px-2 py-1 rounded text-xs"
            >
              {{ factor }}
            </span>
          </div>
        </div>

        <!-- Bearish Factors -->
        <div v-if="signal.factors.bearish?.length > 0" class="bg-red-900/30 rounded-lg p-3">
          <div class="text-red-400 text-sm font-semibold mb-2">? Bearish Factors</div>
          <div class="flex flex-wrap gap-2">
            <span 
              v-for="(factor, idx) in signal.factors.bearish" 
              :key="'bearish-'+idx"
              class="bg-red-600/30 text-red-300 px-2 py-1 rounded text-xs"
            >
              {{ factor }}
            </span>
          </div>
        </div>

        <!-- Skip Reasons -->
        <div v-if="signal.factors.skip_reasons?.length > 0" class="bg-gray-700 rounded-lg p-3">
          <div class="text-gray-400 text-sm font-semibold mb-2">?? Skip Reasons</div>
          <div class="flex flex-wrap gap-2">
            <span 
              v-for="(reason, idx) in signal.factors.skip_reasons" 
              :key="'skip-'+idx"
              class="bg-gray-600 text-gray-300 px-2 py-1 rounded text-xs"
            >
              {{ reason }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Last Updated -->
    <div class="text-xs text-gray-500 text-right">
      Last updated: {{ formatTimestamp(signal?.timestamp) }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

const props = defineProps({
  symbol: {
    type: String,
    default: 'XAUUSDm'
  }
})

const emit = defineEmits(['signal-updated'])

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
const signal = ref(null)
const signalMode = ref('technical')
const isLoading = ref(false)
const selectedSymbol = ref(props.symbol)

// Polling
let pollInterval = null

// Computed
const technicalData = computed(() => {
  if (!signal.value) return null
  
  // Extract technical data from different sources
  const indicators = signal.value.indicators || {}
  const scores = signal.value.scores || {}
  const factors = signal.value.factors || {}
  
  // Try to extract buy/sell score from factors
  let buyScore = 0
  let sellScore = 0
  let session = 'N/A'
  let trend = 'RANGE'
  
  if (factors.bullish) {
    const buyFactor = factors.bullish.find(f => f.includes('Buy Score'))
    if (buyFactor) {
      const match = buyFactor.match(/(\d+)\/10/)
      if (match) buyScore = parseInt(match[1])
    }
    const sessionFactor = factors.bullish.find(f => f.includes('Session'))
    if (sessionFactor) {
      session = sessionFactor.replace('Session: ', '')
    }
    const trendFactor = factors.bullish.find(f => f.includes('Trend'))
    if (trendFactor) {
      trend = trendFactor.replace('Trend: ', '')
    }
  }
  
  if (factors.bearish) {
    const sellFactor = factors.bearish.find(f => f.includes('Sell Score'))
    if (sellFactor) {
      const match = sellFactor.match(/(\d+)\/10/)
      if (match) sellScore = parseInt(match[1])
    }
    const sessionFactor = factors.bearish.find(f => f.includes('Session'))
    if (sessionFactor) {
      session = sessionFactor.replace('Session: ', '')
    }
    const trendFactor = factors.bearish.find(f => f.includes('Trend'))
    if (trendFactor) {
      trend = trendFactor.replace('Trend: ', '')
    }
  }
  
  return {
    buy_score: buyScore || (scores.pattern / 10),
    sell_score: sellScore || 0,
    session: session,
    trend: trend || signal.value.market_regime,
    rsi: indicators.rsi || 50,
    atr: indicators.atr || 0
  }
})

const riskReward = computed(() => {
  if (!signal.value?.risk_management) return 0
  return signal.value.risk_management.risk_reward || 0
})

const signalEmoji = computed(() => {
  const sig = signal.value?.signal
  const map = {
    'STRONG_BUY': '????',
    'BUY': '??',
    'WAIT': '??',
    'SELL': '??',
    'STRONG_SELL': '????'
  }
  return map[sig] || '??'
})

const signalBgClass = computed(() => {
  const sig = signal.value?.signal
  if (sig === 'STRONG_BUY') return 'bg-gradient-to-br from-green-600 to-green-800'
  if (sig === 'BUY') return 'bg-green-700'
  if (sig === 'STRONG_SELL') return 'bg-gradient-to-br from-red-600 to-red-800'
  if (sig === 'SELL') return 'bg-red-700'
  return 'bg-gray-700'
})

const qualityColorClass = computed(() => {
  const q = signal.value?.quality
  if (q === 'PREMIUM') return 'text-purple-400'
  if (q === 'HIGH') return 'text-green-400'
  if (q === 'MEDIUM') return 'text-yellow-400'
  if (q === 'LOW') return 'text-orange-400'
  return 'text-gray-400'
})

const qualityDescription = computed(() => {
  const q = signal.value?.quality
  if (q === 'PREMIUM') return '85%+ confidence - Safest'
  if (q === 'HIGH') return '75%+ confidence - Recommended'
  if (q === 'MEDIUM') return '65%+ confidence - Moderate'
  if (q === 'LOW') return '50%+ confidence - Aggressive'
  return 'Signal conditions not met'
})

const sessionColorClass = computed(() => {
  const s = technicalData.value?.session
  if (s === 'OVERLAP') return 'text-green-400'
  if (s === 'LONDON' || s === 'NY') return 'text-blue-400'
  if (s === 'ASIAN') return 'text-yellow-400'
  return 'text-gray-400'
})

const trendColorClass = computed(() => {
  const t = technicalData.value?.trend
  if (t?.includes('STRONG_UP')) return 'text-green-400'
  if (t?.includes('UP') || t === 'BULLISH') return 'text-green-300'
  if (t?.includes('STRONG_DOWN')) return 'text-red-400'
  if (t?.includes('DOWN') || t === 'BEARISH') return 'text-red-300'
  return 'text-gray-400'
})

const rsiColorClass = computed(() => {
  const rsi = technicalData.value?.rsi || 50
  if (rsi > 70) return 'text-red-400'
  if (rsi < 30) return 'text-green-400'
  return 'text-white'
})

// ?? Display Confidence - Try multiple fields
const displayConfidence = computed(() => {
  if (!signal.value) return 0
  // Try: confidence, enhanced_confidence, base_confidence in order
  const conf = signal.value.confidence 
    || signal.value.enhanced_confidence 
    || signal.value.base_confidence 
    || 0
  return Number(conf) || 0
})

// Methods
function formatPrice(price) {
  if (!price) return '---'
  if (selectedSymbol.value.includes('XAU') || selectedSymbol.value.includes('GOLD')) {
    return '$' + price.toFixed(2)
  }
  return price.toFixed(5)
}

function formatTimestamp(ts) {
  if (!ts) return '---'
  try {
    const date = new Date(ts)
    return date.toLocaleTimeString()
  } catch {
    return ts
  }
}

async function refreshSignal() {
  isLoading.value = true
  try {
    // ?? Use Unified API
    const response = await fetch(`${API_BASE}/api/v1/unified/signal/${selectedSymbol.value}`)
    if (response.ok) {
      const data = await response.json()
      signal.value = data
      signalMode.value = data.signal_mode || 'technical'
      emit('signal-updated', data)
    }
  } catch (error) {
    console.error('Failed to fetch signal:', error)
  } finally {
    isLoading.value = false
  }
}

async function fetchBotStatus() {
  try {
    // ?? Use Unified API
    const response = await fetch(`${API_BASE}/api/v1/unified/status`)
    if (response.ok) {
      const data = await response.json()
      signalMode.value = data.bot?.signal_mode || 'technical'
      
      // Get last signal for selected symbol from unified status
      if (data.signals && data.signals[selectedSymbol.value]) {
        signal.value = data.signals[selectedSymbol.value]
      }
    }
  } catch (error) {
    console.error('Failed to fetch bot status:', error)
  }
}

// Watch for symbol changes
watch(() => props.symbol, (newSymbol) => {
  selectedSymbol.value = newSymbol
  refreshSignal()
})

// Lifecycle
onMounted(() => {
  fetchBotStatus()
  refreshSignal()
  pollInterval = setInterval(refreshSignal, 10000) // Poll every 10 seconds
})

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
})
</script>
