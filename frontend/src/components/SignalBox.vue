<template>
  <div 
    class="card text-center"
    :class="[signalClass, shouldPulse ? pulseClass : '']"
  >
    <!-- Signal Icon -->
    <div class="mb-4">
      <div 
        class="w-20 h-20 mx-auto rounded-full flex items-center justify-center text-4xl"
        :class="iconBgClass"
      >
        {{ signalIcon }}
      </div>
    </div>
    
    <!-- Signal Text -->
    <h2 class="text-3xl font-bold mb-2">{{ signalText }}</h2>
    
    <!-- Confidence -->
    <div class="text-5xl font-bold mb-4" :class="confidenceColor">
      {{ confidence }}%
    </div>
    <p class="text-gray-400 mb-4">Confidence</p>
    
    <!-- Signal Duration Countdown -->
    <div v-if="hasDuration && !isExpired" class="mb-4 p-4 rounded-lg" :class="durationBgClass">
      <div class="flex items-center justify-center gap-2 mb-2">
        <span class="text-lg">⏱️</span>
        <span class="text-2xl font-mono font-bold" :class="countdownColor">
          {{ formattedCountdown }}
        </span>
      </div>
      <div class="text-sm" :class="durationTextClass">
        {{ durationStatusText }}
      </div>
      <!-- Progress Bar -->
      <div class="mt-2 w-full bg-gray-700 rounded-full h-2">
        <div 
          class="h-2 rounded-full transition-all duration-1000"
          :class="progressBarClass"
          :style="{ width: progressPercent + '%' }"
        ></div>
      </div>
      <!-- Strength Badge -->
      <div class="mt-2">
        <span class="px-2 py-1 rounded text-xs font-semibold" :class="strengthBadgeClass">
          {{ strengthText }}
        </span>
      </div>
    </div>
    
    <!-- Expired Signal Warning -->
    <div v-else-if="hasDuration && isExpired" class="mb-4 p-3 bg-red-900/30 border border-red-500/50 rounded-lg">
      <div class="flex items-center justify-center gap-2 text-red-400">
        <span>⚠️</span>
        <span class="font-semibold">สัญญาณหมดอายุแล้ว!</span>
      </div>
      <p class="text-red-300 text-sm mt-1">รอสัญญาณใหม่</p>
    </div>
    
    <!-- Vote Summary -->
    <div class="flex justify-center space-x-8 mb-4">
      <div class="text-center">
        <div class="text-2xl font-bold text-green-500">{{ bullishVotes }}</div>
        <div class="text-gray-500 text-sm">Bullish</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-red-500">{{ bearishVotes }}</div>
        <div class="text-gray-500 text-sm">Bearish</div>
      </div>
    </div>
    
    <!-- Timestamp -->
    <p class="text-gray-500 text-sm">
      {{ timestamp }}
    </p>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'

const props = defineProps({
  signal: {
    type: Object,
    default: null
  }
})

// Countdown timer
const remainingSeconds = ref(0)
let countdownInterval = null

// Calculate remaining time from signal duration
const calculateRemaining = () => {
  if (!props.signal?.duration?.expires_at) return 0
  const expiresAt = new Date(props.signal.duration.expires_at)
  const now = new Date()
  return Math.max(0, Math.floor((expiresAt - now) / 1000))
}

// Start countdown timer
const startCountdown = () => {
  if (countdownInterval) clearInterval(countdownInterval)
  remainingSeconds.value = calculateRemaining()
  
  countdownInterval = setInterval(() => {
    remainingSeconds.value = calculateRemaining()
    if (remainingSeconds.value <= 0) {
      clearInterval(countdownInterval)
    }
  }, 1000)
}

// Watch for signal changes
watch(() => props.signal, (newSignal) => {
  if (newSignal?.duration) {
    startCountdown()
  }
}, { immediate: true })

onMounted(() => {
  if (props.signal?.duration) {
    startCountdown()
  }
})

onUnmounted(() => {
  if (countdownInterval) clearInterval(countdownInterval)
})

// Duration computed properties
const hasDuration = computed(() => {
  return props.signal?.duration && props.signal.signal !== 'WAIT'
})

const isExpired = computed(() => {
  return remainingSeconds.value <= 0
})

const isWarning = computed(() => {
  if (!props.signal?.duration) return false
  const totalSeconds = props.signal.duration.estimated_minutes * 60
  return remainingSeconds.value < totalSeconds * 0.5
})

const isCritical = computed(() => {
  if (!props.signal?.duration) return false
  const totalSeconds = props.signal.duration.estimated_minutes * 60
  return remainingSeconds.value < totalSeconds * 0.2
})

const formattedCountdown = computed(() => {
  const secs = remainingSeconds.value
  if (secs <= 0) return '00:00'
  
  const hours = Math.floor(secs / 3600)
  const mins = Math.floor((secs % 3600) / 60)
  const sec = secs % 60
  
  if (hours > 0) {
    return `${hours}:${mins.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
  }
  return `${mins.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
})

const progressPercent = computed(() => {
  if (!props.signal?.duration?.estimated_minutes) return 0
  const totalSeconds = props.signal.duration.estimated_minutes * 60
  return Math.max(0, (remainingSeconds.value / totalSeconds) * 100)
})

const durationStatusText = computed(() => {
  if (isExpired.value) return 'สัญญาณหมดอายุแล้ว'
  if (isCritical.value) return '⚡ เหลือเวลาน้อยมาก! เทรดเดี๋ยวนี้!'
  if (isWarning.value) return '⚠️ เหลือเวลาไม่มาก'
  return `สัญญาณอยู่ได้อีกประมาณ ${props.signal?.duration?.estimated_minutes || 0} นาที`
})

const strengthText = computed(() => {
  const strength = props.signal?.duration?.strength
  switch (strength) {
    case 'VERY_STRONG': return '💪 แข็งแกร่งมาก'
    case 'STRONG': return '✊ แข็งแกร่ง'
    case 'MODERATE': return '👌 ปานกลาง'
    case 'WEAK': return '👋 อ่อน'
    default: return ''
  }
})

const countdownColor = computed(() => {
  if (isCritical.value) return 'text-red-400 animate-pulse'
  if (isWarning.value) return 'text-yellow-400'
  return 'text-green-400'
})

const durationBgClass = computed(() => {
  if (isCritical.value) return 'bg-red-900/20 border border-red-500/30'
  if (isWarning.value) return 'bg-yellow-900/20 border border-yellow-500/30'
  return 'bg-green-900/20 border border-green-500/30'
})

const durationTextClass = computed(() => {
  if (isCritical.value) return 'text-red-300'
  if (isWarning.value) return 'text-yellow-300'
  return 'text-green-300'
})

const progressBarClass = computed(() => {
  if (isCritical.value) return 'bg-red-500'
  if (isWarning.value) return 'bg-yellow-500'
  return 'bg-green-500'
})

const strengthBadgeClass = computed(() => {
  const strength = props.signal?.duration?.strength
  switch (strength) {
    case 'VERY_STRONG': return 'bg-green-600 text-white'
    case 'STRONG': return 'bg-green-500/80 text-white'
    case 'MODERATE': return 'bg-yellow-500/80 text-gray-900'
    case 'WEAK': return 'bg-gray-500/80 text-white'
    default: return 'bg-gray-600 text-white'
  }
})

const signalText = computed(() => {
  if (!props.signal) return 'WAIT'
  return props.signal.signal.replace('_', ' ')
})

const confidence = computed(() => {
  if (!props.signal) return 0
  return Math.round(props.signal.confidence)
})

const bullishVotes = computed(() => {
  return props.signal?.vote_details?.bullish || 0
})

const bearishVotes = computed(() => {
  return props.signal?.vote_details?.bearish || 0
})

const timestamp = computed(() => {
  if (!props.signal?.timestamp) return ''
  return new Date(props.signal.timestamp).toLocaleString()
})

const signalClass = computed(() => {
  if (!props.signal) return ''
  const signal = props.signal.signal.toLowerCase().replace('_', '-')
  return `signal-${signal}`
})

const signalIcon = computed(() => {
  if (!props.signal) return '⏸️'
  switch (props.signal.signal) {
    case 'STRONG_BUY': return '🚀'
    case 'BUY': return '📈'
    case 'SELL': return '📉'
    case 'STRONG_SELL': return '💥'
    default: return '⏸️'
  }
})

const iconBgClass = computed(() => {
  if (!props.signal) return 'bg-gray-800'
  switch (props.signal.signal) {
    case 'STRONG_BUY':
    case 'BUY':
      return 'bg-green-900/50'
    case 'SELL':
    case 'STRONG_SELL':
      return 'bg-red-900/50'
    default:
      return 'bg-yellow-900/50'
  }
})

const confidenceColor = computed(() => {
  if (confidence.value >= 80) return 'text-green-500'
  if (confidence.value >= 70) return 'text-yellow-500'
  return 'text-gray-400'
})

const shouldPulse = computed(() => {
  return props.signal && props.signal.signal !== 'WAIT' && confidence.value >= 70
})

const pulseClass = computed(() => {
  if (!props.signal) return ''
  return ['STRONG_BUY', 'BUY'].includes(props.signal.signal) ? 'pulse-buy' : 'pulse-sell'
})
</script>
