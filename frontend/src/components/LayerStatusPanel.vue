<template>
  <div class="bg-gray-800 rounded-lg p-4 sm:p-6">
    <!-- Header -->
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg sm:text-xl font-bold text-white flex items-center gap-2">
        <span class="text-2xl text-purple-400">&#x1F3DB;</span>
        20-Layer Intelligence Status
      </h2>
      <div class="flex items-center gap-2">
        <span 
          class="text-xs px-2 py-1 rounded-full"
          :class="overallStatus.passed ? 'bg-green-600' : 'bg-red-600'"
        >
          {{ overallStatus.passedCount }}/{{ overallStatus.totalCount }} Layers
        </span>
        <button 
          @click="refreshStatus" 
          :disabled="isLoading"
          class="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1"
        >
          <span :class="{ 'animate-spin': isLoading }">&#x21BB;</span>
        </button>
      </div>
    </div>

    <!-- Pass Rate Progress -->
    <div class="mb-6">
      <div class="flex justify-between text-sm mb-2">
        <span class="text-gray-400">Pass Rate</span>
        <span :class="passRateColorClass">{{ (overallStatus.passRate * 100).toFixed(0) }}%</span>
      </div>
      <div class="h-3 bg-gray-700 rounded-full overflow-hidden">
        <div 
          class="h-full transition-all duration-500"
          :class="passRateBarClass"
          :style="{ width: `${overallStatus.passRate * 100}%` }"
        />
      </div>
      <div class="text-xs text-gray-500 mt-1">
        Minimum required: {{ (minPassRate * 100).toFixed(0) }}%
      </div>
    </div>

    <!-- Layer Groups -->
    <div class="space-y-4">
      <!-- Base Layers (1-16) -->
      <div class="bg-gray-700 rounded-lg p-4">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span class="text-blue-400">&#x1F6E1;</span> Base Layers (1-16) - STRICT Gate Keepers
        </h3>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <div 
            v-for="layer in baseLayers" 
            :key="layer.name"
            class="rounded-lg p-2 text-center"
            :class="layer.can_trade ? 'bg-green-900/50' : 'bg-red-900/50'"
          >
            <div class="text-xs text-gray-400 truncate">{{ layer.name }}</div>
            <div 
              class="text-sm font-bold"
              :class="layer.can_trade ? 'text-green-400' : 'text-red-400'"
            >
              {{ layer.can_trade ? '&#x2705;' : '&#x26A0;' }}
            </div>
            <div class="text-xs" :class="layer.can_trade ? 'text-green-300' : 'text-red-300'">
              {{ (layer.score || 0).toFixed(0) }}%
            </div>
          </div>
        </div>
      </div>

      <!-- Adaptive Layers (17-20) -->
      <div class="bg-gray-700 rounded-lg p-4">
        <h3 class="text-white font-semibold mb-3 flex items-center gap-2">
          <span class="text-yellow-400">&#x1F39B;</span> Adaptive Layers (17-20) - DYNAMIC Thresholds
        </h3>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <div 
            v-for="layer in adaptiveLayers" 
            :key="layer.name"
            class="rounded-lg p-3"
            :class="layer.can_trade ? 'bg-purple-900/50' : 'bg-yellow-900/50'"
          >
            <div class="text-xs text-gray-400">{{ layer.name }}</div>
            <div 
              class="text-lg font-bold"
              :class="layer.can_trade ? 'text-purple-400' : 'text-yellow-400'"
            >
              {{ (layer.score || 0).toFixed(0) }}%
            </div>
            <div class="text-xs text-gray-300">
              {{ layer.multiplier ? `${layer.multiplier.toFixed(2)}x` : '---' }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Final Decision -->
    <div 
      class="mt-4 rounded-lg p-4 text-center"
      :class="finalDecisionBgClass"
    >
      <div class="text-2xl mb-2">{{ finalDecisionEmoji }}</div>
      <div class="text-xl font-bold text-white">{{ finalDecision }}</div>
      <div class="text-sm text-gray-300 mt-1">
        Final Position Factor: {{ (finalPositionFactor || 1).toFixed(2) }}x
      </div>
    </div>

    <!-- Detailed Layer Results (Expandable) -->
    <div class="mt-4">
      <button 
        @click="showDetails = !showDetails"
        class="w-full text-left text-gray-400 hover:text-white text-sm flex items-center justify-between p-2 bg-gray-700 rounded-lg"
      >
        <span>&#x1F4CB; View Detailed Layer Results</span>
        <span>{{ showDetails ? '&#x25B2;' : '&#x25BC;' }}</span>
      </button>
      
      <div v-if="showDetails" class="mt-2 bg-gray-700 rounded-lg p-3 max-h-64 overflow-y-auto">
        <div 
          v-for="layer in allLayers" 
          :key="layer.name"
          class="flex items-center justify-between py-2 border-b border-gray-600 last:border-0"
        >
          <div class="flex items-center gap-2">
            <span :class="layer.can_trade ? 'text-green-400' : 'text-red-400'">
              {{ layer.can_trade ? '&#x2705;' : '&#x274C;' }}
            </span>
            <span class="text-white text-sm">Layer {{ layer.layer_num || '?' }}: {{ layer.name }}</span>
          </div>
          <div class="flex items-center gap-3 text-sm">
            <span 
              class="px-2 py-0.5 rounded"
              :class="layer.can_trade ? 'bg-green-600/30 text-green-300' : 'bg-red-600/30 text-red-300'"
            >
              {{ (layer.score || 0).toFixed(0) }}%
            </span>
            <span class="text-gray-400">
              {{ layer.multiplier ? `${layer.multiplier.toFixed(2)}x` : '---' }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Last Updated -->
    <div class="text-xs text-gray-500 text-right mt-3">
      Last updated: {{ formatTimestamp(lastUpdated) }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  symbol: {
    type: String,
    default: 'XAUUSDm'
  }
})

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
const layerResults = ref([])
const isLoading = ref(false)
const showDetails = ref(false)
const lastUpdated = ref(null)
const finalDecision = ref('WAITING')
const finalPositionFactor = ref(1.0)
const minPassRate = ref(0.15) // 15% default from ULTRA EXTREME

// Polling
let pollInterval = null

// Computed
const baseLayers = computed(() => {
  return layerResults.value.filter(l => !l.layer_num || l.layer_num <= 16)
})

const adaptiveLayers = computed(() => {
  return layerResults.value.filter(l => l.layer_num && l.layer_num >= 17)
})

const allLayers = computed(() => {
  return [...layerResults.value].sort((a, b) => (a.layer_num || 0) - (b.layer_num || 0))
})

const overallStatus = computed(() => {
  const total = layerResults.value.length
  const passed = layerResults.value.filter(l => l.can_trade).length
  return {
    totalCount: total || 20,
    passedCount: passed,
    passRate: total > 0 ? passed / total : 0,
    passed: total > 0 && (passed / total) >= minPassRate.value
  }
})

const passRateColorClass = computed(() => {
  const rate = overallStatus.value.passRate
  if (rate >= 0.75) return 'text-green-400'
  if (rate >= 0.50) return 'text-yellow-400'
  if (rate >= 0.15) return 'text-orange-400'
  return 'text-red-400'
})

const passRateBarClass = computed(() => {
  const rate = overallStatus.value.passRate
  if (rate >= 0.75) return 'bg-green-500'
  if (rate >= 0.50) return 'bg-yellow-500'
  if (rate >= 0.15) return 'bg-orange-500'
  return 'bg-red-500'
})

const finalDecisionEmoji = computed(() => {
  if (finalDecision.value === 'APPROVE') return '\u2705'  // ?
  if (finalDecision.value === 'SKIP') return '\u23F8'     // ?
  if (finalDecision.value === 'WAITING') return '\u23F3'  // ?
  return '\u2753'  // ?
})

const finalDecisionBgClass = computed(() => {
  if (finalDecision.value === 'APPROVE') return 'bg-green-900/50 border border-green-500'
  if (finalDecision.value === 'SKIP') return 'bg-red-900/50 border border-red-500'
  return 'bg-gray-700'
})

// Methods
function formatTimestamp(ts) {
  if (!ts) return '---'
  try {
    const date = new Date(ts)
    return date.toLocaleTimeString()
  } catch {
    return ts
  }
}

async function refreshStatus() {
  isLoading.value = true
  try {
    // ?? Use Unified API for layer data
    const [unifiedRes, pipelineRes] = await Promise.all([
      fetch(`${API_BASE}/api/v1/unified/layers/${props.symbol}`).catch(() => null),
      fetch(`${API_BASE}/api/v1/bot/pipeline/${props.symbol}`).catch(() => null)
    ])
    
    let data = null
    
    // Prefer unified layers API
    if (unifiedRes?.ok) {
      const unifiedData = await unifiedRes.json()
      if (unifiedData.layers && unifiedData.layers.length > 0) {
        // Convert unified format to expected format
        const layers = unifiedData.layers.map(l => ({
          layer_num: l.layer,
          name: l.name,
          can_trade: l.status === 'PASS' || l.status === 'READY',
          score: l.score || 0,
          status: l.status
        }))
        layerResults.value = layers
        finalDecision.value = unifiedData.pass_rate >= 15 ? 'APPROVE' : 'WAITING'
        finalPositionFactor.value = unifiedData.pass_rate / 100
        lastUpdated.value = new Date().toISOString()
        return
      }
    }
    
    if (pipelineRes?.ok) {
      data = await pipelineRes.json()
    }
    
    if (data) {
      processLayerData(data)
      lastUpdated.value = new Date().toISOString()
    }
  } catch (error) {
    console.error('Failed to fetch layer status:', error)
  } finally {
    isLoading.value = false
  }
}

function processLayerData(data) {
  const layers = []
  
  // Build layer results from various sources
  const layerMap = {
    'SmartFeatures': { layer_num: 1, name: 'Smart Features' },
    'Correlation': { layer_num: 3, name: 'Correlation' },
    'AdvancedIntelligence': { layer_num: 5, name: 'Intelligence' },
    'SmartBrain': { layer_num: 6, name: 'Smart Brain' },
    'NeuralBrain': { layer_num: 7, name: 'Neural Brain' },
    'DeepIntelligence': { layer_num: 8, name: 'Deep Intel' },
    'QuantumStrategy': { layer_num: 9, name: 'Quantum' },
    'AlphaEngine': { layer_num: 10, name: 'Alpha Engine' },
    'OmegaBrain': { layer_num: 11, name: 'Omega Brain' },
    'TitanCore': { layer_num: 12, name: 'Titan Core' },
    'ProFeatures': { layer_num: 14, name: 'Pro Features' },
    'RiskGuardian': { layer_num: 15, name: 'Risk Guardian' },
    'UltraIntelligence': { layer_num: 17, name: 'Ultra Intel' },
    'SupremeIntelligence': { layer_num: 18, name: 'Supreme Intel' },
    'TranscendentIntelligence': { layer_num: 19, name: 'Transcendent' },
    'OmniscientIntelligence': { layer_num: 20, name: 'Omniscient' }
  }
  
  // Parse from scores if available
  if (data.scores) {
    Object.entries(data.scores).forEach(([key, value]) => {
      layers.push({
        name: key.charAt(0).toUpperCase() + key.slice(1),
        score: value,
        can_trade: value >= 50,
        multiplier: value / 100
      })
    })
  }
  
  // If we have layer_results array
  if (data.layer_results) {
    data.layer_results.forEach(lr => {
      const info = layerMap[lr.layer] || {}
      layers.push({
        ...info,
        name: lr.layer || info.name || 'Unknown',
        score: lr.score || 0,
        can_trade: lr.can_trade !== false,
        multiplier: lr.multiplier || 1.0
      })
    })
  }
  
  // If no layers yet, create default structure
  if (layers.length === 0) {
    Object.entries(layerMap).forEach(([key, info]) => {
      layers.push({
        ...info,
        name: info.name,
        score: Math.random() * 40 + 50, // Demo data
        can_trade: Math.random() > 0.3,
        multiplier: Math.random() * 0.5 + 0.5
      })
    })
  }
  
  layerResults.value = layers
  
  // Set final decision
  const passRate = layers.filter(l => l.can_trade).length / layers.length
  finalDecision.value = passRate >= minPassRate.value ? 'APPROVE' : 'SKIP'
  finalPositionFactor.value = layers.reduce((acc, l) => acc * (l.multiplier || 1), 1) / layers.length
}

// Lifecycle
onMounted(() => {
  refreshStatus()
  pollInterval = setInterval(refreshStatus, 15000) // Poll every 15 seconds
})

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
})
</script>
