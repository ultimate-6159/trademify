<template>
  <div 
    class="factor-card rounded-xl p-4 transition-all duration-300 hover:scale-[1.02] cursor-pointer"
    :class="[cardBgClass, borderClass]"
    @click="expanded = !expanded"
  >
    <div class="flex items-center justify-between">
      <!-- Left side: Icon & Title -->
      <div class="flex items-center gap-3">
        <div 
          class="w-12 h-12 rounded-lg flex items-center justify-center text-2xl"
          :class="iconBgClass"
        >
          {{ icon }}
        </div>
        <div>
          <h5 class="text-white font-semibold">{{ title }}</h5>
          <p class="text-gray-400 text-sm">{{ description }}</p>
        </div>
      </div>
      
      <!-- Right side: Score & Status -->
      <div class="flex items-center gap-4">
        <!-- Score Display -->
        <div class="text-right">
          <div class="text-3xl font-bold" :class="scoreColorClass">
            {{ formattedScore }}
          </div>
          <div class="text-xs" :class="statusColorClass">
            {{ statusLabel }}
          </div>
        </div>
        
        <!-- Status Icon -->
        <div 
          class="w-10 h-10 rounded-full flex items-center justify-center text-xl"
          :class="statusBgClass"
        >
          {{ statusIcon }}
        </div>
      </div>
    </div>
    
    <!-- Progress Bar -->
    <div class="mt-4">
      <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div 
          class="h-full rounded-full transition-all duration-700 ease-out"
          :class="progressColorClass"
          :style="{ width: `${score || 0}%` }"
        ></div>
      </div>
      
      <!-- Scale Labels -->
      <div class="flex justify-between mt-1 text-xs text-gray-500">
        <span>0</span>
        <span>40</span>
        <span>60</span>
        <span>80</span>
        <span>100</span>
      </div>
    </div>
    
    <!-- Expanded Details -->
    <div v-if="expanded && details" class="mt-4 pt-4 border-t border-gray-600">
      <div class="text-gray-300 text-sm">
        <span class="text-gray-400">Details:</span> {{ details }}
      </div>
      
      <!-- Interpretation Guide -->
      <div class="mt-3 bg-gray-800/50 rounded-lg p-3">
        <div class="text-xs text-gray-400 mb-2">📖 การอ่านค่า:</div>
        <div class="grid grid-cols-2 gap-2 text-xs">
          <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded bg-green-500"></span>
            <span class="text-gray-300">80-100: ดีเยี่ยม</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded bg-blue-500"></span>
            <span class="text-gray-300">60-80: ดี</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded bg-yellow-500"></span>
            <span class="text-gray-300">40-60: ปานกลาง</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded bg-red-500"></span>
            <span class="text-gray-300">0-40: ต้องระวัง</span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Expand indicator -->
    <div class="text-center mt-2">
      <span class="text-gray-500 text-xs">
        {{ expanded ? '▲ คลิกเพื่อซ่อน' : '▼ คลิกเพื่อดูรายละเอียด' }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  title: { type: String, required: true },
  icon: { type: String, default: '📊' },
  score: { type: Number, default: 0 },
  description: { type: String, default: '' },
  status: { type: String, default: 'neutral' }, // excellent, good, moderate, weak, neutral
  details: { type: String, default: '' },
  color: { type: String, default: 'blue' } // purple, blue, cyan, green, orange, pink, teal
})

const expanded = ref(false)

const formattedScore = computed(() => {
  if (props.score === null || props.score === undefined) return '--'
  return props.score.toFixed(0)
})

// Card background based on status
const cardBgClass = computed(() => {
  switch (props.status) {
    case 'excellent': return 'bg-gradient-to-br from-green-900/40 to-gray-800'
    case 'good': return 'bg-gradient-to-br from-blue-900/40 to-gray-800'
    case 'moderate': return 'bg-gradient-to-br from-yellow-900/40 to-gray-800'
    case 'weak': return 'bg-gradient-to-br from-red-900/40 to-gray-800'
    default: return 'bg-gray-800'
  }
})

// Border color based on status
const borderClass = computed(() => {
  switch (props.status) {
    case 'excellent': return 'border border-green-500/30'
    case 'good': return 'border border-blue-500/30'
    case 'moderate': return 'border border-yellow-500/30'
    case 'weak': return 'border border-red-500/30'
    default: return 'border border-gray-700'
  }
})

// Icon background based on color prop
const iconBgClass = computed(() => {
  const colorMap = {
    purple: 'bg-purple-500/20',
    blue: 'bg-blue-500/20',
    cyan: 'bg-cyan-500/20',
    green: 'bg-green-500/20',
    orange: 'bg-orange-500/20',
    pink: 'bg-pink-500/20',
    teal: 'bg-teal-500/20'
  }
  return colorMap[props.color] || 'bg-gray-700'
})

// Score color
const scoreColorClass = computed(() => {
  if (props.score >= 80) return 'text-green-400'
  if (props.score >= 60) return 'text-blue-400'
  if (props.score >= 40) return 'text-yellow-400'
  return 'text-red-400'
})

// Status label
const statusLabel = computed(() => {
  switch (props.status) {
    case 'excellent': return 'ดีเยี่ยม'
    case 'good': return 'ดี'
    case 'moderate': return 'ปานกลาง'
    case 'weak': return 'ต้องระวัง'
    default: return 'กลาง'
  }
})

// Status color
const statusColorClass = computed(() => {
  switch (props.status) {
    case 'excellent': return 'text-green-400'
    case 'good': return 'text-blue-400'
    case 'moderate': return 'text-yellow-400'
    case 'weak': return 'text-red-400'
    default: return 'text-gray-400'
  }
})

// Status background
const statusBgClass = computed(() => {
  switch (props.status) {
    case 'excellent': return 'bg-green-500/20 text-green-400'
    case 'good': return 'bg-blue-500/20 text-blue-400'
    case 'moderate': return 'bg-yellow-500/20 text-yellow-400'
    case 'weak': return 'bg-red-500/20 text-red-400'
    default: return 'bg-gray-700 text-gray-400'
  }
})

// Status icon
const statusIcon = computed(() => {
  switch (props.status) {
    case 'excellent': return '✓'
    case 'good': return '✓'
    case 'moderate': return '−'
    case 'weak': return '!'
    default: return '?'
  }
})

// Progress bar color
const progressColorClass = computed(() => {
  if (props.score >= 80) return 'bg-gradient-to-r from-green-500 to-emerald-400'
  if (props.score >= 60) return 'bg-gradient-to-r from-blue-500 to-cyan-400'
  if (props.score >= 40) return 'bg-gradient-to-r from-yellow-500 to-orange-400'
  return 'bg-gradient-to-r from-red-500 to-red-400'
})
</script>

<style scoped>
.factor-card {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.factor-card:hover {
  box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}
</style>
