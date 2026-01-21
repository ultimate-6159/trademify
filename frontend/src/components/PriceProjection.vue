<template>
  <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
    <!-- Current Price -->
    <div class="bg-dark-200 rounded-lg p-4">
      <div class="text-gray-400 text-sm mb-1">Current Price</div>
      <div class="text-2xl font-bold text-white">{{ formatPrice(projection.current) }}</div>
    </div>
    
    <!-- Projected Price -->
    <div class="bg-dark-200 rounded-lg p-4">
      <div class="text-gray-400 text-sm mb-1">Projected Price</div>
      <div class="text-2xl font-bold" :class="projectedColor">
        {{ formatPrice(projection.projected) }}
        <span class="text-sm">
          ({{ projectedChange }})
        </span>
      </div>
    </div>
    
    <!-- Stop Loss -->
    <div class="bg-dark-200 rounded-lg p-4">
      <div class="text-gray-400 text-sm mb-1">Stop Loss</div>
      <div class="text-2xl font-bold text-red-500">{{ formatPrice(projection.stop_loss) }}</div>
      <div class="text-gray-500 text-sm">{{ stopLossDistance }}</div>
    </div>
    
    <!-- Take Profit -->
    <div class="bg-dark-200 rounded-lg p-4">
      <div class="text-gray-400 text-sm mb-1">Take Profit</div>
      <div class="text-2xl font-bold text-green-500">{{ formatPrice(projection.take_profit) }}</div>
      <div class="text-gray-500 text-sm">{{ takeProfitDistance }}</div>
    </div>
  </div>
  
  <!-- Risk/Reward Ratio -->
  <div class="mt-4 bg-dark-200 rounded-lg p-4">
    <div class="flex items-center justify-between">
      <div>
        <div class="text-gray-400 text-sm mb-1">Risk/Reward Ratio</div>
        <div class="text-2xl font-bold text-primary">1:{{ riskRewardRatio }}</div>
      </div>
      
      <!-- Visual bar -->
      <div class="flex-1 mx-8">
        <div class="relative h-4 bg-dark-100 rounded-full overflow-hidden">
          <!-- SL Zone -->
          <div 
            class="absolute left-0 top-0 h-full bg-red-500/30"
            :style="{ width: stopLossWidth + '%' }"
          ></div>
          
          <!-- Current position -->
          <div 
            class="absolute top-0 h-full w-1 bg-white"
            :style="{ left: currentPosWidth + '%' }"
          ></div>
          
          <!-- TP Zone -->
          <div 
            class="absolute right-0 top-0 h-full bg-green-500/30"
            :style="{ width: takeProfitWidth + '%' }"
          ></div>
        </div>
        
        <div class="flex justify-between text-xs text-gray-500 mt-1">
          <span>SL: {{ formatPrice(projection.stop_loss) }}</span>
          <span>Entry: {{ formatPrice(projection.current) }}</span>
          <span>TP: {{ formatPrice(projection.take_profit) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  projection: {
    type: Object,
    default: () => ({
      current: 0,
      projected: 0,
      stop_loss: 0,
      take_profit: 0
    })
  }
})

function formatPrice(price) {
  if (!price) return '-'
  return price.toFixed(5)
}

const projectedColor = computed(() => {
  if (!props.projection.projected || !props.projection.current) return 'text-white'
  return props.projection.projected > props.projection.current ? 'text-green-500' : 'text-red-500'
})

const projectedChange = computed(() => {
  if (!props.projection.projected || !props.projection.current) return '0%'
  const change = ((props.projection.projected - props.projection.current) / props.projection.current) * 100
  const sign = change >= 0 ? '+' : ''
  return `${sign}${change.toFixed(2)}%`
})

const stopLossDistance = computed(() => {
  if (!props.projection.stop_loss || !props.projection.current) return '-'
  const distance = Math.abs((props.projection.stop_loss - props.projection.current) / props.projection.current) * 100
  return `-${distance.toFixed(2)}%`
})

const takeProfitDistance = computed(() => {
  if (!props.projection.take_profit || !props.projection.current) return '-'
  const distance = Math.abs((props.projection.take_profit - props.projection.current) / props.projection.current) * 100
  return `+${distance.toFixed(2)}%`
})

const riskRewardRatio = computed(() => {
  if (!props.projection.stop_loss || !props.projection.take_profit || !props.projection.current) return '0.0'
  
  const risk = Math.abs(props.projection.current - props.projection.stop_loss)
  const reward = Math.abs(props.projection.take_profit - props.projection.current)
  
  if (risk === 0) return '∞'
  return (reward / risk).toFixed(1)
})

const stopLossWidth = computed(() => {
  if (!props.projection.stop_loss || !props.projection.take_profit || !props.projection.current) return 20
  
  const range = props.projection.take_profit - props.projection.stop_loss
  const slRange = props.projection.current - props.projection.stop_loss
  
  return (slRange / range) * 100
})

const currentPosWidth = computed(() => {
  return stopLossWidth.value
})

const takeProfitWidth = computed(() => {
  return 100 - stopLossWidth.value
})
</script>
