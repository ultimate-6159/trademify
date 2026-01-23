<template>
  <div 
    class="connection-status"
    :class="statusClass"
    @click="toggleDetails"
  >
    <!-- Status Indicator -->
    <div class="status-dot" :class="dotClass"></div>
    
    <!-- Status Text -->
    <span class="status-text">{{ statusText }}</span>
    
    <!-- Reconnecting Spinner -->
    <svg v-if="status.reconnecting" class="spinner" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" fill="none" stroke-dasharray="31.4" stroke-linecap="round">
        <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
      </circle>
    </svg>

    <!-- Details Panel -->
    <transition name="slide">
      <div v-if="showDetails" class="details-panel">
        <div class="detail-row">
          <span class="label">สถานะ:</span>
          <span :class="healthClass">{{ healthText }}</span>
        </div>
        <div class="detail-row">
          <span class="label">เชื่อมต่อล่าสุด:</span>
          <span>{{ lastSuccessText }}</span>
        </div>
        <div class="detail-row" v-if="status.consecutiveErrors > 0">
          <span class="label">Errors ติดต่อกัน:</span>
          <span class="text-red-400">{{ status.consecutiveErrors }}</span>
        </div>
        <div class="detail-row">
          <span class="label">API URL:</span>
          <span class="text-xs text-gray-400 truncate">{{ apiUrl }}</span>
        </div>
        
        <!-- Actions -->
        <div class="actions" v-if="!status.isOnline">
          <button @click.stop="reconnect" class="reconnect-btn" :disabled="status.reconnecting">
            {{ status.reconnecting ? 'กำลังเชื่อมต่อ...' : '🔄 เชื่อมต่อใหม่' }}
          </button>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import api, { onConnectionChange, getConnectionStatus } from '@/services/api'

const showDetails = ref(false)
const status = ref(getConnectionStatus())

let unsubscribe = null

onMounted(() => {
  unsubscribe = onConnectionChange((newStatus) => {
    status.value = newStatus
  })
})

onUnmounted(() => {
  if (unsubscribe) unsubscribe()
})

const apiUrl = computed(() => api.getBaseUrl())

const statusClass = computed(() => ({
  'online': status.value.isOnline && !status.value.reconnecting,
  'offline': !status.value.isOnline && !status.value.reconnecting,
  'reconnecting': status.value.reconnecting
}))

const dotClass = computed(() => ({
  'dot-green': status.value.isOnline && status.value.consecutiveErrors === 0,
  'dot-yellow': status.value.isOnline && status.value.consecutiveErrors > 0,
  'dot-red': !status.value.isOnline,
  'dot-pulse': status.value.reconnecting
}))

const statusText = computed(() => {
  if (status.value.reconnecting) return 'กำลังเชื่อมต่อ...'
  if (!status.value.isOnline) return 'ออฟไลน์'
  if (status.value.consecutiveErrors > 0) return 'ไม่เสถียร'
  return 'เชื่อมต่อแล้ว'
})

const healthClass = computed(() => ({
  'text-green-400': status.value.health === 'excellent',
  'text-yellow-400': status.value.health === 'degraded',
  'text-red-400': status.value.health === 'critical'
}))

const healthText = computed(() => ({
  'excellent': '✅ ดีเยี่ยม',
  'degraded': '⚠️ ไม่เสถียร',
  'critical': '❌ วิกฤต'
}[status.value.health] || 'ไม่ทราบ'))

const lastSuccessText = computed(() => {
  if (!status.value.lastSuccessTime) return 'ยังไม่เคย'
  const diff = Date.now() - status.value.lastSuccessTime
  if (diff < 5000) return 'เมื่อสักครู่'
  if (diff < 60000) return `${Math.floor(diff / 1000)} วินาทีที่แล้ว`
  if (diff < 3600000) return `${Math.floor(diff / 60000)} นาทีที่แล้ว`
  return 'นานมากแล้ว'
})

const toggleDetails = () => {
  showDetails.value = !showDetails.value
}

const reconnect = async () => {
  await api.reconnect()
}
</script>

<style scoped>
.connection-status {
  @apply relative flex items-center gap-2 px-3 py-1.5 rounded-full cursor-pointer transition-all;
  @apply bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700;
  font-size: 0.75rem;
}

.connection-status.online {
  @apply border-green-500/30;
}

.connection-status.offline {
  @apply border-red-500/50 bg-red-900/20;
}

.connection-status.reconnecting {
  @apply border-yellow-500/50 bg-yellow-900/20;
}

.status-dot {
  @apply w-2 h-2 rounded-full;
}

.dot-green {
  @apply bg-green-500;
  box-shadow: 0 0 6px rgba(34, 197, 94, 0.6);
}

.dot-yellow {
  @apply bg-yellow-500;
  box-shadow: 0 0 6px rgba(234, 179, 8, 0.6);
}

.dot-red {
  @apply bg-red-500;
  box-shadow: 0 0 6px rgba(239, 68, 68, 0.6);
}

.dot-pulse {
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(1.2); }
}

.status-text {
  @apply text-gray-300 font-medium;
}

.spinner {
  @apply w-4 h-4 text-yellow-400;
}

.details-panel {
  @apply absolute top-full left-0 mt-2 p-3 rounded-lg z-50;
  @apply bg-gray-900 border border-gray-700 shadow-xl;
  min-width: 220px;
}

.detail-row {
  @apply flex justify-between items-center py-1 text-xs;
}

.label {
  @apply text-gray-500;
}

.actions {
  @apply mt-3 pt-2 border-t border-gray-700;
}

.reconnect-btn {
  @apply w-full px-3 py-1.5 rounded text-xs font-medium;
  @apply bg-blue-600 hover:bg-blue-500 text-white;
  @apply disabled:opacity-50 disabled:cursor-not-allowed;
  @apply transition-colors;
}

.slide-enter-active,
.slide-leave-active {
  transition: all 0.2s ease;
}

.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>
