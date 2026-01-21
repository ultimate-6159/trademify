<template>
  <div class="factor-breakdown space-y-4">
    <!-- Overall Score Summary -->
    <div class="bg-gradient-to-r from-gray-700 to-gray-800 rounded-xl p-4 border border-gray-600">
      <div class="flex items-center justify-between mb-3">
        <h4 class="text-white font-semibold flex items-center gap-2">
          <span class="text-xl">📊</span>
          Factor Analysis Summary
        </h4>
        <div class="text-2xl font-bold" :class="getOverallScoreColor(finalScore)">
          {{ finalScore?.toFixed(0) }}%
        </div>
      </div>
      
      <!-- Overall Progress Bar -->
      <div class="relative h-4 bg-gray-600 rounded-full overflow-hidden">
        <div 
          class="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
          :class="getOverallBarColor(finalScore)"
          :style="{ width: `${finalScore || 0}%` }"
        ></div>
        <!-- Threshold Markers -->
        <div class="absolute inset-y-0 left-[60%] w-0.5 bg-yellow-500/50" title="Medium (60%)"></div>
        <div class="absolute inset-y-0 left-[75%] w-0.5 bg-green-500/50" title="High (75%)"></div>
        <div class="absolute inset-y-0 left-[85%] w-0.5 bg-purple-500/50" title="Premium (85%)"></div>
      </div>
      <div class="flex justify-between mt-1 text-xs text-gray-400">
        <span>0%</span>
        <span>60% Medium</span>
        <span>75% High</span>
        <span>85% Premium</span>
        <span>100%</span>
      </div>
    </div>

    <!-- Detailed Factor Cards -->
    <div class="grid grid-cols-1 gap-3">
      <!-- Pattern Match Factor -->
      <FactorCard 
        title="Pattern Match"
        icon="🎯"
        :score="factors?.pattern || scores?.pattern"
        :description="getPatternDescription(factors?.pattern || scores?.pattern)"
        :status="getFactorStatus('pattern', factors?.pattern || scores?.pattern)"
        :details="getFactorDetails('pattern')"
        color="purple"
      />
      
      <!-- Trend Alignment Factor -->
      <FactorCard 
        title="Trend Alignment"
        icon="📈"
        :score="factors?.trend || scores?.trend"
        :description="getTrendDescription(factors?.trend || scores?.trend)"
        :status="getFactorStatus('trend', factors?.trend || scores?.trend)"
        :details="getFactorDetails('trend')"
        color="blue"
      />
      
      <!-- Volume Confirmation Factor -->
      <FactorCard 
        title="Volume Confirmation"
        icon="📊"
        :score="factors?.volume || scores?.volume"
        :description="getVolumeDescription(factors?.volume || scores?.volume)"
        :status="getFactorStatus('volume', factors?.volume || scores?.volume)"
        :details="getFactorDetails('volume')"
        color="cyan"
      />
      
      <!-- Momentum Factor -->
      <FactorCard 
        title="Momentum"
        icon="⚡"
        :score="factors?.momentum || scores?.momentum"
        :description="getMomentumDescription(factors?.momentum || scores?.momentum, indicators)"
        :status="getFactorStatus('momentum', factors?.momentum || scores?.momentum)"
        :details="getMomentumDetails(indicators)"
        color="orange"
      />
      
      <!-- Session Timing Factor -->
      <FactorCard 
        title="Session Timing"
        icon="🕐"
        :score="factors?.session || scores?.session"
        :description="getSessionDescription(factors?.session || scores?.session)"
        :status="getFactorStatus('session', factors?.session || scores?.session)"
        :details="getFactorDetails('session')"
        color="green"
      />
      
      <!-- Volatility Factor -->
      <FactorCard 
        title="Volatility"
        icon="🌊"
        :score="factors?.volatility || scores?.volatility"
        :description="getVolatilityDescription(factors?.volatility || scores?.volatility)"
        :status="getFactorStatus('volatility', factors?.volatility || scores?.volatility)"
        :details="getFactorDetails('volatility')"
        color="pink"
      />
      
      <!-- Pattern Recency Factor -->
      <FactorCard 
        title="Pattern Recency"
        icon="📅"
        :score="factors?.recency || scores?.recency"
        :description="getRecencyDescription(factors?.recency || scores?.recency)"
        :status="getFactorStatus('recency', factors?.recency || scores?.recency)"
        :details="getFactorDetails('recency')"
        color="teal"
      />
    </div>

    <!-- Technical Indicators Panel -->
    <div v-if="indicators" class="bg-gray-700/50 rounded-xl p-4 border border-gray-600">
      <h4 class="text-white font-semibold mb-3 flex items-center gap-2">
        <span class="text-xl">🔬</span>
        Technical Indicators
      </h4>
      
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <!-- RSI -->
        <div class="bg-gray-800 rounded-lg p-3 text-center">
          <div class="text-gray-400 text-xs mb-1">RSI (14)</div>
          <div 
            class="text-2xl font-bold"
            :class="getRsiColor(indicators.rsi)"
          >
            {{ indicators.rsi?.toFixed(1) || '---' }}
          </div>
          <div class="text-xs mt-1" :class="getRsiLabelColor(indicators.rsi)">
            {{ getRsiLabel(indicators.rsi) }}
          </div>
        </div>
        
        <!-- MACD -->
        <div class="bg-gray-800 rounded-lg p-3 text-center">
          <div class="text-gray-400 text-xs mb-1">MACD Trend</div>
          <div 
            class="text-lg font-bold"
            :class="getMacdColor(indicators.macd_trend)"
          >
            {{ indicators.macd_trend || '---' }}
          </div>
          <div class="text-xs mt-1 text-gray-400">
            {{ getMacdSignal(indicators.macd_histogram) }}
          </div>
        </div>
        
        <!-- EMA Trend -->
        <div class="bg-gray-800 rounded-lg p-3 text-center">
          <div class="text-gray-400 text-xs mb-1">EMA Trend</div>
          <div 
            class="text-lg font-bold"
            :class="getTrendColorClass(indicators.ema_trend)"
          >
            {{ indicators.ema_trend || '---' }}
          </div>
          <div class="text-xs mt-1 text-gray-400">
            Fast vs Slow EMA
          </div>
        </div>
        
        <!-- ATR / Volatility -->
        <div class="bg-gray-800 rounded-lg p-3 text-center">
          <div class="text-gray-400 text-xs mb-1">ATR %</div>
          <div 
            class="text-2xl font-bold"
            :class="getAtrColor(indicators.atr_percent)"
          >
            {{ indicators.atr_percent?.toFixed(2) || '---' }}%
          </div>
          <div class="text-xs mt-1 text-gray-400">
            {{ getAtrLabel(indicators.atr_percent) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Reasons Panel -->
    <div v-if="reasons" class="space-y-2">
      <!-- Bullish Reasons -->
      <div v-if="reasons.bullish?.length" class="bg-green-900/30 border border-green-700/50 rounded-lg p-3">
        <div class="text-green-400 font-semibold text-sm mb-2 flex items-center gap-2">
          <span>✅</span> Bullish Factors ({{ reasons.bullish.length }})
        </div>
        <ul class="space-y-1">
          <li 
            v-for="(reason, idx) in reasons.bullish" 
            :key="idx"
            class="text-green-300 text-sm flex items-start gap-2"
          >
            <span class="text-green-500 mt-0.5">•</span>
            <span>{{ reason }}</span>
          </li>
        </ul>
      </div>
      
      <!-- Bearish Reasons -->
      <div v-if="reasons.bearish?.length" class="bg-red-900/30 border border-red-700/50 rounded-lg p-3">
        <div class="text-red-400 font-semibold text-sm mb-2 flex items-center gap-2">
          <span>⚠️</span> Bearish Factors ({{ reasons.bearish.length }})
        </div>
        <ul class="space-y-1">
          <li 
            v-for="(reason, idx) in reasons.bearish" 
            :key="idx"
            class="text-red-300 text-sm flex items-start gap-2"
          >
            <span class="text-red-500 mt-0.5">•</span>
            <span>{{ reason }}</span>
          </li>
        </ul>
      </div>
      
      <!-- Skip Reasons -->
      <div v-if="reasons.skip?.length" class="bg-yellow-900/30 border border-yellow-700/50 rounded-lg p-3">
        <div class="text-yellow-400 font-semibold text-sm mb-2 flex items-center gap-2">
          <span>⏸️</span> Caution / Skip Reasons ({{ reasons.skip.length }})
        </div>
        <ul class="space-y-1">
          <li 
            v-for="(reason, idx) in reasons.skip" 
            :key="idx"
            class="text-yellow-300 text-sm flex items-start gap-2"
          >
            <span class="text-yellow-500 mt-0.5">•</span>
            <span>{{ reason }}</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

// Child component for individual factor
import FactorCard from './FactorCard.vue'

const props = defineProps({
  finalScore: { type: Number, default: 0 },
  factors: { type: Object, default: null }, // Detailed factors array
  scores: { type: Object, default: null },  // Simple scores object
  indicators: { type: Object, default: null },
  reasons: { type: Object, default: null },
  factorDetails: { type: Array, default: () => [] } // Full factor analysis array
})

// Helper functions
function getOverallScoreColor(score) {
  if (!score) return 'text-gray-400'
  if (score >= 85) return 'text-purple-400'
  if (score >= 75) return 'text-green-400'
  if (score >= 60) return 'text-yellow-400'
  return 'text-red-400'
}

function getOverallBarColor(score) {
  if (!score) return 'bg-gray-500'
  if (score >= 85) return 'bg-gradient-to-r from-purple-500 to-pink-500'
  if (score >= 75) return 'bg-gradient-to-r from-green-500 to-emerald-400'
  if (score >= 60) return 'bg-gradient-to-r from-yellow-500 to-orange-400'
  return 'bg-gradient-to-r from-red-500 to-red-400'
}

function getFactorStatus(name, score) {
  if (!score || score === 0) return 'neutral'
  if (score >= 80) return 'excellent'
  if (score >= 60) return 'good'
  if (score >= 40) return 'moderate'
  return 'weak'
}

function getFactorDetails(name) {
  const detail = props.factorDetails?.find(f => f.name?.toLowerCase().includes(name))
  return detail?.details || ''
}

// Pattern descriptions
function getPatternDescription(score) {
  if (!score) return 'ไม่มีข้อมูล Pattern'
  if (score >= 85) return 'พบ Pattern ที่ตรงกันมาก (Strong Match)'
  if (score >= 70) return 'พบ Pattern ที่ตรงกันดี (Good Match)'
  if (score >= 50) return 'พบ Pattern ที่ตรงกันปานกลาง (Moderate)'
  return 'Pattern ไม่ชัดเจน (Weak Match)'
}

// Trend descriptions
function getTrendDescription(score) {
  if (!score) return 'ไม่มีข้อมูล Trend'
  if (score >= 80) return 'สัญญาณสอดคล้องกับ Trend หลัก (Aligned)'
  if (score >= 60) return 'Trend สนับสนุนทิศทาง'
  if (score >= 40) return 'Trend ไม่ชัดเจน (Sideways)'
  return '⚠️ สวนทาง Trend (Counter-trend)'
}

// Volume descriptions
function getVolumeDescription(score) {
  if (!score) return 'ไม่มีข้อมูล Volume'
  if (score >= 80) return 'Volume ยืนยันสัญญาณ (Volume Spike)'
  if (score >= 60) return 'Volume ปกติ สนับสนุนสัญญาณ'
  if (score >= 40) return 'Volume ต่ำกว่าค่าเฉลี่ย'
  return '⚠️ Volume ต่ำมาก - ขาดความเชื่อมั่น'
}

// Momentum descriptions
function getMomentumDescription(score, indicators) {
  if (!score) return 'ไม่มีข้อมูล Momentum'
  const rsi = indicators?.rsi || 50
  if (score >= 80) return `Momentum ยืนยันทุกตัว (RSI: ${rsi.toFixed(0)})`
  if (score >= 60) return `Momentum สนับสนุนบางส่วน (RSI: ${rsi.toFixed(0)})`
  if (score >= 40) return `Momentum ไม่ชัดเจน (RSI: ${rsi.toFixed(0)})`
  return `⚠️ Momentum สวนทาง (RSI: ${rsi.toFixed(0)})`
}

function getMomentumDetails(indicators) {
  if (!indicators) return ''
  const parts = []
  if (indicators.rsi) {
    const rsiStatus = indicators.rsi > 70 ? 'Overbought' : indicators.rsi < 30 ? 'Oversold' : 'Neutral'
    parts.push(`RSI ${indicators.rsi.toFixed(1)} (${rsiStatus})`)
  }
  if (indicators.macd_trend) {
    parts.push(`MACD ${indicators.macd_trend}`)
  }
  return parts.join(' | ')
}

// Session descriptions
function getSessionDescription(score) {
  if (!score) return 'ไม่มีข้อมูล Session'
  if (score >= 90) return '🌟 Prime Time - London/NY Overlap'
  if (score >= 70) return '✅ Session ที่ดี - ตลาดคึกคัก'
  if (score >= 50) return 'Session ปกติ'
  if (score >= 30) return '⚠️ Session เงียบ - Liquidity ต่ำ'
  return '❌ ตลาดปิด หรือ Weekend'
}

// Volatility descriptions
function getVolatilityDescription(score) {
  if (!score) return 'ไม่มีข้อมูล Volatility'
  if (score >= 80) return '✅ Volatility เหมาะสม - สภาพตลาดดี'
  if (score >= 50) return 'Volatility ปานกลาง'
  if (score >= 30) return '⚠️ Volatility ต่ำ - ตลาดเงียบ'
  return '⚠️ Volatility สูงมาก - ความเสี่ยงสูง'
}

// Recency descriptions
function getRecencyDescription(score) {
  if (!score) return 'ไม่มีข้อมูล Pattern Date'
  if (score >= 70) return '✅ Pattern ล่าสุด - น่าเชื่อถือ'
  if (score >= 50) return 'Pattern ค่อนข้างใหม่'
  if (score >= 30) return 'Pattern เก่า - ต้องระวัง'
  return '⚠️ Pattern เก่ามาก - อาจล้าสมัย'
}

// RSI helpers
function getRsiColor(rsi) {
  if (!rsi) return 'text-gray-400'
  if (rsi > 70) return 'text-red-400'
  if (rsi < 30) return 'text-green-400'
  return 'text-blue-400'
}

function getRsiLabelColor(rsi) {
  if (!rsi) return 'text-gray-500'
  if (rsi > 70) return 'text-red-400'
  if (rsi < 30) return 'text-green-400'
  return 'text-gray-400'
}

function getRsiLabel(rsi) {
  if (!rsi) return '---'
  if (rsi > 80) return '⚠️ Overbought มาก'
  if (rsi > 70) return 'Overbought'
  if (rsi < 20) return '⚠️ Oversold มาก'
  if (rsi < 30) return 'Oversold'
  if (rsi > 50) return 'Bullish Zone'
  return 'Bearish Zone'
}

// MACD helpers
function getMacdColor(trend) {
  if (!trend) return 'text-gray-400'
  if (trend === 'BULLISH') return 'text-green-400'
  if (trend === 'BEARISH') return 'text-red-400'
  return 'text-gray-400'
}

function getMacdSignal(histogram) {
  if (!histogram) return '---'
  if (histogram > 0) return 'Histogram Positive ↑'
  return 'Histogram Negative ↓'
}

// Trend helpers
function getTrendColorClass(trend) {
  if (!trend) return 'text-gray-400'
  if (trend === 'UP' || trend === 'BULLISH') return 'text-green-400'
  if (trend === 'DOWN' || trend === 'BEARISH') return 'text-red-400'
  return 'text-yellow-400'
}

// ATR helpers
function getAtrColor(atr) {
  if (!atr) return 'text-gray-400'
  if (atr > 3) return 'text-red-400'
  if (atr < 0.5) return 'text-yellow-400'
  return 'text-green-400'
}

function getAtrLabel(atr) {
  if (!atr) return '---'
  if (atr > 3) return 'สูงมาก - เสี่ยง'
  if (atr > 1.5) return 'สูง'
  if (atr < 0.3) return 'ต่ำมาก'
  if (atr < 0.5) return 'ต่ำ'
  return 'ปกติ'
}
</script>

<style scoped>
.factor-breakdown {
  animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
