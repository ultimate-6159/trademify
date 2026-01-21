<template>
  <div class="space-y-2 max-h-64 overflow-y-auto">
    <div v-if="patterns.length === 0" class="text-gray-500 text-center py-4">
      No matched patterns found
    </div>
    
    <div 
      v-for="(pattern, index) in patterns" 
      :key="index"
      class="flex items-center justify-between p-3 bg-dark-200 rounded-lg"
    >
      <div class="flex items-center space-x-3">
        <div class="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold text-sm">
          {{ index + 1 }}
        </div>
        <div>
          <div class="text-white font-medium">Pattern #{{ pattern.index }}</div>
          <div class="text-gray-500 text-sm">
            Correlation: {{ formatCorrelation(pattern.correlation) }}
          </div>
        </div>
      </div>
      
      <div class="text-right">
        <div class="text-sm">
          <span 
            class="inline-block w-2 h-2 rounded-full mr-1"
            :class="getCorrelationClass(pattern.correlation)"
          ></span>
          {{ getCorrelationLabel(pattern.correlation) }}
        </div>
        <div class="text-gray-500 text-xs">
          Distance: {{ pattern.distance?.toFixed(4) || 'N/A' }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  patterns: {
    type: Array,
    default: () => []
  }
})

function formatCorrelation(corr) {
  if (corr === undefined || corr === null) return 'N/A'
  return (corr * 100).toFixed(1) + '%'
}

function getCorrelationClass(corr) {
  if (corr >= 0.95) return 'bg-green-500'
  if (corr >= 0.90) return 'bg-green-400'
  if (corr >= 0.85) return 'bg-yellow-500'
  return 'bg-red-500'
}

function getCorrelationLabel(corr) {
  if (corr >= 0.95) return 'Excellent'
  if (corr >= 0.90) return 'Very Good'
  if (corr >= 0.85) return 'Good'
  return 'Moderate'
}
</script>
