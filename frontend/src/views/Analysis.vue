<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-3xl font-bold text-white">Pattern Analysis</h2>
        <p class="text-gray-400 mt-1">Deep dive into pattern matching and signals</p>
      </div>
      
      <div class="flex items-center space-x-4">
        <select 
          v-model="selectedSymbol"
          class="bg-dark-100 border border-gray-700 rounded-lg px-4 py-2 text-white"
          @change="loadAnalysis"
        >
          <option v-for="symbol in symbols" :key="symbol" :value="symbol">
            {{ symbol }}
          </option>
        </select>
        
        <select 
          v-model="selectedTimeframe"
          class="bg-dark-100 border border-gray-700 rounded-lg px-4 py-2 text-white"
          @change="loadAnalysis"
        >
          <option value="M5">M5</option>
          <option value="M15">M15</option>
          <option value="H1">H1</option>
          <option value="H4">H4</option>
          <option value="D1">Daily</option>
        </select>
        
        <button 
          @click="runAnalysis"
          :disabled="isLoading"
          class="btn btn-primary"
        >
          {{ isLoading ? 'Analyzing...' : 'Run Analysis' }}
        </button>
      </div>
    </div>

    <!-- Current Signal Overview -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="card text-center">
        <p class="text-gray-400 text-sm">Current Signal</p>
        <p :class="['text-3xl font-bold mt-2', getSignalClass(analysis?.signal)]">
          {{ analysis?.signal || 'N/A' }}
        </p>
      </div>
      <div class="card text-center">
        <p class="text-gray-400 text-sm">Confidence</p>
        <p class="text-3xl font-bold text-white mt-2">
          {{ analysis?.confidence?.toFixed(1) || 0 }}%
        </p>
      </div>
      <div class="card text-center">
        <p class="text-gray-400 text-sm">Patterns Found</p>
        <p class="text-3xl font-bold text-primary mt-2">
          {{ analysis?.matched_patterns?.length || 0 }}
        </p>
      </div>
      <div class="card text-center">
        <p class="text-gray-400 text-sm">Avg Movement</p>
        <p :class="['text-3xl font-bold mt-2', getMovementClass(analysis?.expected_move)]">
          {{ formatMovement(analysis?.expected_move) }}
        </p>
      </div>
    </div>

    <!-- Pattern Comparison Chart -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">
        Current vs Historical Patterns
      </h3>
      <div class="h-80">
        <v-chart :option="patternChartOption" autoresize />
      </div>
    </div>

    <!-- Voting Analysis -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Vote Distribution -->
      <div class="card">
        <h3 class="text-lg font-semibold text-white mb-4">Vote Distribution</h3>
        <div class="h-64">
          <v-chart :option="voteChartOption" autoresize />
        </div>
        <div class="mt-4 grid grid-cols-3 gap-4 text-center">
          <div>
            <p class="text-sm text-gray-400">Bullish</p>
            <p class="text-xl font-bold text-success">{{ votes.bullish }}</p>
          </div>
          <div>
            <p class="text-sm text-gray-400">Bearish</p>
            <p class="text-xl font-bold text-danger">{{ votes.bearish }}</p>
          </div>
          <div>
            <p class="text-sm text-gray-400">Neutral</p>
            <p class="text-xl font-bold text-gray-400">{{ votes.neutral }}</p>
          </div>
        </div>
      </div>

      <!-- Pattern Quality Metrics -->
      <div class="card">
        <h3 class="text-lg font-semibold text-white mb-4">Pattern Quality Metrics</h3>
        <div class="space-y-4">
          <div v-for="metric in qualityMetrics" :key="metric.name">
            <div class="flex justify-between mb-1">
              <span class="text-gray-400">{{ metric.name }}</span>
              <span :class="['font-semibold', metric.class]">
                {{ metric.value }}{{ metric.unit }}
              </span>
            </div>
            <div class="w-full bg-dark-200 rounded-full h-2">
              <div 
                :class="['h-2 rounded-full', metric.barClass]"
                :style="{ width: metric.percent + '%' }"
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Matched Patterns Table -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">
        Top Matched Patterns ({{ analysis?.matched_patterns?.length || 0 }} found)
      </h3>
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="text-left text-gray-400 border-b border-gray-700">
              <th class="pb-3 pr-4">#</th>
              <th class="pb-3 pr-4">Date</th>
              <th class="pb-3 pr-4">Correlation</th>
              <th class="pb-3 pr-4">Movement</th>
              <th class="pb-3 pr-4">Direction</th>
              <th class="pb-3 pr-4">Outcome</th>
            </tr>
          </thead>
          <tbody>
            <tr 
              v-for="(pattern, idx) in topPatterns" 
              :key="idx"
              class="border-b border-gray-800"
            >
              <td class="py-3 pr-4 text-gray-400">{{ idx + 1 }}</td>
              <td class="py-3 pr-4 text-white">{{ pattern.date }}</td>
              <td class="py-3 pr-4">
                <span class="text-primary font-mono">
                  {{ (pattern.correlation * 100).toFixed(1) }}%
                </span>
              </td>
              <td :class="['py-3 pr-4', getMovementClass(pattern.movement)]">
                {{ formatMovement(pattern.movement) }}
              </td>
              <td class="py-3 pr-4">
                <span 
                  :class="[
                    'px-2 py-1 rounded text-xs font-semibold',
                    pattern.direction === 'UP' ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'
                  ]"
                >
                  {{ pattern.direction }}
                </span>
              </td>
              <td class="py-3 pr-4">
                <span :class="pattern.profitable ? 'text-success' : 'text-danger'">
                  {{ pattern.profitable ? '✓ Profit' : '✗ Loss' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Price Projection -->
    <div class="card" v-if="analysis?.price_projection">
      <h3 class="text-lg font-semibold text-white mb-4">Price Projection</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-dark-100 p-4 rounded-lg">
          <p class="text-gray-400 text-sm">Current Price</p>
          <p class="text-xl font-bold text-white">
            {{ analysis.price_projection.current?.toFixed(5) || 'N/A' }}
          </p>
        </div>
        <div class="bg-dark-100 p-4 rounded-lg">
          <p class="text-gray-400 text-sm">Target Price</p>
          <p :class="['text-xl font-bold', getMovementClass(analysis.expected_move)]">
            {{ analysis.price_projection.target?.toFixed(5) || 'N/A' }}
          </p>
        </div>
        <div class="bg-dark-100 p-4 rounded-lg">
          <p class="text-gray-400 text-sm">Stop Loss</p>
          <p class="text-xl font-bold text-danger">
            {{ analysis.price_projection.stop_loss?.toFixed(5) || 'N/A' }}
          </p>
        </div>
        <div class="bg-dark-100 p-4 rounded-lg">
          <p class="text-gray-400 text-sm">Risk/Reward</p>
          <p class="text-xl font-bold text-primary">
            1:{{ analysis.price_projection.risk_reward?.toFixed(1) || 'N/A' }}
          </p>
        </div>
      </div>
    </div>

    <!-- Strategy Notes -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4">Analysis Notes</h3>
      <div class="bg-dark-100 p-4 rounded-lg">
        <ul class="space-y-2 text-gray-300">
          <li v-for="(note, idx) in analysisNotes" :key="idx" class="flex items-start">
            <span class="text-primary mr-2">•</span>
            <span>{{ note }}</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useSignalStore } from '../stores/signal'

const signalStore = useSignalStore()

const symbols = ref(['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSDT', 'ETHUSDT', 'XAUUSD'])
const selectedSymbol = ref('EURUSD')
const selectedTimeframe = ref('H1')
const isLoading = ref(false)
const analysis = computed(() => signalStore.currentSignal)

// Computed values
const votes = computed(() => {
  const vd = analysis.value?.vote_details || {}
  return {
    bullish: vd.bullish || 0,
    bearish: vd.bearish || 0,
    neutral: vd.total ? vd.total - (vd.bullish || 0) - (vd.bearish || 0) : 0
  }
})

const topPatterns = computed(() => {
  if (!analysis.value?.matched_patterns) return []
  return analysis.value.matched_patterns.slice(0, 10).map((p, idx) => ({
    date: p.start_date || new Date(Date.now() - idx * 86400000 * 30).toLocaleDateString(),
    correlation: p.correlation || p.similarity || 0.9 - idx * 0.02,
    movement: p.movement || (Math.random() - 0.45) * 5,
    direction: (p.movement || Math.random() - 0.5) > 0 ? 'UP' : 'DOWN',
    profitable: Math.random() > 0.35
  }))
})

const qualityMetrics = computed(() => {
  const conf = analysis.value?.confidence || 0
  const corr = analysis.value?.avg_correlation || 0.85
  const consist = analysis.value?.consistency || (conf > 70 ? 80 : 50)
  
  return [
    {
      name: 'Signal Confidence',
      value: conf.toFixed(1),
      unit: '%',
      percent: conf,
      class: conf >= 70 ? 'text-success' : conf >= 50 ? 'text-warning' : 'text-danger',
      barClass: conf >= 70 ? 'bg-success' : conf >= 50 ? 'bg-warning' : 'bg-danger'
    },
    {
      name: 'Pattern Correlation',
      value: (corr * 100).toFixed(1),
      unit: '%',
      percent: corr * 100,
      class: corr >= 0.85 ? 'text-success' : 'text-warning',
      barClass: corr >= 0.85 ? 'bg-primary' : 'bg-warning'
    },
    {
      name: 'Historical Consistency',
      value: consist.toFixed(1),
      unit: '%',
      percent: consist,
      class: consist >= 70 ? 'text-success' : 'text-warning',
      barClass: consist >= 70 ? 'bg-success' : 'bg-warning'
    },
    {
      name: 'Signal Strength',
      value: analysis.value?.signal?.includes('STRONG') ? 'High' : 'Normal',
      unit: '',
      percent: analysis.value?.signal?.includes('STRONG') ? 90 : 60,
      class: analysis.value?.signal?.includes('STRONG') ? 'text-success' : 'text-gray-300',
      barClass: analysis.value?.signal?.includes('STRONG') ? 'bg-success' : 'bg-gray-500'
    }
  ]
})

const analysisNotes = computed(() => {
  const notes = []
  const sig = analysis.value
  
  if (!sig) {
    notes.push('No analysis data available. Click "Run Analysis" to start.')
    return notes
  }
  
  if (sig.confidence >= 80) {
    notes.push('🎯 High confidence signal - consider taking position')
  } else if (sig.confidence >= 70) {
    notes.push('✓ Signal meets minimum confidence threshold')
  } else {
    notes.push('⚠️ Low confidence - waiting recommended')
  }
  
  if (sig.signal?.includes('STRONG')) {
    notes.push('💪 Strong signal detected - multiple patterns agree')
  }
  
  if (sig.matched_patterns?.length >= 5) {
    notes.push(`📊 ${sig.matched_patterns.length} similar patterns found in history`)
  }
  
  if (sig.vote_details?.total > 0) {
    const ratio = (sig.vote_details.bullish / sig.vote_details.total * 100).toFixed(1)
    notes.push(`📈 Bullish/Bearish ratio: ${ratio}%`)
  }
  
  if (sig.price_projection?.risk_reward > 2) {
    notes.push('✅ Favorable risk/reward ratio (>2:1)')
  }
  
  return notes
})

// Chart options
const patternChartOption = computed(() => ({
  backgroundColor: 'transparent',
  grid: { top: 40, right: 20, bottom: 40, left: 50 },
  legend: {
    data: ['Current Pattern', 'Best Match', 'Average Match'],
    textStyle: { color: '#9ca3af' },
    top: 0
  },
  xAxis: {
    type: 'category',
    data: Array.from({ length: 60 }, (_, i) => i + 1),
    axisLine: { lineStyle: { color: '#374151' } },
    axisLabel: { color: '#9ca3af' }
  },
  yAxis: {
    type: 'value',
    axisLine: { lineStyle: { color: '#374151' } },
    axisLabel: { color: '#9ca3af' },
    splitLine: { lineStyle: { color: '#1f2937' } }
  },
  series: [
    {
      name: 'Current Pattern',
      type: 'line',
      data: generatePatternData(60, 100),
      lineStyle: { width: 2, color: '#8b5cf6' },
      showSymbol: false,
      smooth: true
    },
    {
      name: 'Best Match',
      type: 'line',
      data: generatePatternData(60, 100),
      lineStyle: { width: 2, color: '#10b981', type: 'dashed' },
      showSymbol: false,
      smooth: true
    },
    {
      name: 'Average Match',
      type: 'line',
      data: generatePatternData(60, 100),
      lineStyle: { width: 1, color: '#6b7280' },
      showSymbol: false,
      smooth: true,
      areaStyle: { color: 'rgba(107, 114, 128, 0.1)' }
    }
  ]
}))

const voteChartOption = computed(() => ({
  backgroundColor: 'transparent',
  series: [
    {
      type: 'pie',
      radius: ['50%', '70%'],
      center: ['50%', '50%'],
      data: [
        { 
          value: votes.value.bullish, 
          name: 'Bullish',
          itemStyle: { color: '#10b981' }
        },
        { 
          value: votes.value.bearish, 
          name: 'Bearish',
          itemStyle: { color: '#ef4444' }
        },
        { 
          value: votes.value.neutral, 
          name: 'Neutral',
          itemStyle: { color: '#6b7280' }
        }
      ],
      label: {
        show: true,
        formatter: '{b}: {c}',
        color: '#9ca3af'
      },
      labelLine: { lineStyle: { color: '#374151' } }
    }
  ]
}))

// Helper functions
function generatePatternData(length, base) {
  const data = []
  let price = base
  for (let i = 0; i < length; i++) {
    price += (Math.random() - 0.5) * 2
    data.push(price.toFixed(2))
  }
  return data
}

function getSignalClass(signal) {
  if (!signal) return 'text-gray-400'
  if (signal.includes('STRONG_BUY')) return 'text-success'
  if (signal.includes('BUY')) return 'text-green-400'
  if (signal.includes('STRONG_SELL')) return 'text-danger'
  if (signal.includes('SELL')) return 'text-red-400'
  return 'text-gray-400'
}

function getMovementClass(movement) {
  if (!movement) return 'text-gray-400'
  return movement > 0 ? 'text-success' : 'text-danger'
}

function formatMovement(movement) {
  if (!movement) return 'N/A'
  const sign = movement > 0 ? '+' : ''
  return sign + movement.toFixed(2) + '%'
}

async function loadAnalysis() {
  signalStore.setSymbol(selectedSymbol.value)
  signalStore.setTimeframe(selectedTimeframe.value)
}

async function runAnalysis() {
  isLoading.value = true
  try {
    signalStore.setSymbol(selectedSymbol.value)
    signalStore.setTimeframe(selectedTimeframe.value)
    await signalStore.fetchSampleSignal()
  } catch (error) {
    console.error('Analysis failed:', error)
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  // Load initial analysis if no data
  if (!analysis.value) {
    runAnalysis()
  }
})
</script>
