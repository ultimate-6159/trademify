<template>
  <div class="connection-status" :class="statusClass" @click="toggleDetails">
    <!-- Status Indicator -->
    <div class="status-dot" :class="dotClass"></div>

    <!-- Status Text -->
    <span class="status-text">{{ statusText }}</span>

    <!-- Quality Badge -->
    <span v-if="status.isOnline" class="quality-badge" :class="qualityClass">
      {{ qualityText }}
    </span>

    <!-- Latency -->
    <span v-if="status.avgLatency > 0" class="latency"> {{ status.avgLatency }}ms </span>

    <!-- Reconnecting Spinner -->
    <svg v-if="status.reconnecting" class="spinner" viewBox="0 0 24 24">
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        stroke-width="3"
        fill="none"
        stroke-dasharray="31.4"
        stroke-linecap="round"
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from="0 12 12"
          to="360 12 12"
          dur="1s"
          repeatCount="indefinite"
        />
      </circle>
    </svg>

    <!-- Details Panel -->
    <transition name="slide">
      <div v-if="showDetails" class="details-panel">
        <!-- Connection Info -->
        <div class="section">
          <div class="section-title">📡 การเชื่อมต่อ</div>
          <div class="detail-row">
            <span class="label">สถานะ:</span>
            <span :class="healthClass">{{ healthText }}</span>
          </div>
          <div class="detail-row">
            <span class="label">คุณภาพ:</span>
            <span :class="qualityClass">{{ qualityTextFull }}</span>
          </div>
          <div class="detail-row">
            <span class="label">Latency เฉลี่ย:</span>
            <span>{{ status.avgLatency || 0 }}ms</span>
          </div>
          <div class="detail-row">
            <span class="label">Timeout:</span>
            <span>{{ Math.round((status.currentTimeout || 10000) / 1000) }}s</span>
          </div>
        </div>

        <!-- Circuit Breaker -->
        <div class="section">
          <div class="section-title">🛡️ Circuit Breaker</div>
          <div class="detail-row">
            <span class="label">State:</span>
            <span :class="circuitClass">{{ status.circuitState || 'CLOSED' }}</span>
          </div>
          <div class="detail-row" v-if="status.consecutiveErrors > 0">
            <span class="label">Errors:</span>
            <span class="text-red-400">{{ status.consecutiveErrors }}</span>
          </div>
        </div>

        <!-- Performance -->
        <div class="section" v-if="metrics">
          <div class="section-title">⚡ Performance</div>
          <div class="detail-row">
            <span class="label">Success Rate:</span>
            <span :class="successRateClass">{{ metrics.successRate }}%</span>
          </div>
          <div class="detail-row">
            <span class="label">Cache Hit:</span>
            <span class="text-blue-400">{{ metrics.cacheHitRate }}%</span>
          </div>
          <div class="detail-row">
            <span class="label">Total Requests:</span>
            <span>{{ metrics.totalRequests }}</span>
          </div>
        </div>

        <!-- Queues -->
        <div class="section">
          <div class="section-title">📦 Queues</div>
          <div class="detail-row">
            <span class="label">Cache:</span>
            <span>{{ status.cacheSize || 0 }} items</span>
          </div>
          <div class="detail-row">
            <span class="label">Offline Queue:</span>
            <span :class="(status.offlineQueueSize || 0) > 0 ? 'text-yellow-400' : ''">
              {{ status.offlineQueueSize || 0 }} pending
            </span>
          </div>
        </div>

        <!-- API URL -->
        <div class="section">
          <div class="section-title">🌐 API</div>
          <div class="api-url">{{ apiUrl }}</div>
        </div>

        <!-- Actions -->
        <div class="actions">
          <button
            @click.stop="reconnect"
            class="action-btn primary"
            :disabled="status.reconnecting"
          >
            {{ status.reconnecting ? 'กำลังเชื่อมต่อ...' : '🔄 เชื่อมต่อใหม่' }}
          </button>
          <button @click.stop="clearCache" class="action-btn">🧹 ล้าง Cache</button>
          <button @click.stop="exportMetrics" class="action-btn">📤 Export</button>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import api, { onConnectionChange, getConnectionStatus, getPerformanceMetrics } from "@/services/api";

const showDetails = ref(false);
const status = ref(getConnectionStatus());
const metrics = ref(null);

let unsubscribe = null;
let metricsInterval = null;

onMounted(() => {
  unsubscribe = onConnectionChange((newStatus) => {
    status.value = newStatus;
  });

  // Update metrics every 5 seconds
  const updateMetrics = () => {
    try {
      metrics.value = getPerformanceMetrics();
    } catch (e) {
      console.warn('Failed to get metrics');
    }
  };
  updateMetrics();
  metricsInterval = setInterval(updateMetrics, 5000);
});

onUnmounted(() => {
  if (unsubscribe) unsubscribe();
  if (metricsInterval) clearInterval(metricsInterval);
});

const apiUrl = computed(() => api.getBaseUrl());

const statusClass = computed(() => ({
  online: status.value.isOnline && !status.value.reconnecting,
  offline: !status.value.isOnline && !status.value.reconnecting,
  reconnecting: status.value.reconnecting,
}));

const dotClass = computed(() => ({
  "dot-green": status.value.isOnline && status.value.consecutiveErrors === 0,
  "dot-yellow": status.value.isOnline && status.value.consecutiveErrors > 0,
  "dot-red": !status.value.isOnline,
  "dot-pulse": status.value.reconnecting,
}));

const statusText = computed(() => {
  if (status.value.reconnecting) return "กำลังเชื่อมต่อ...";
  if (!status.value.isOnline) return "ออฟไลน์";
  if (status.value.consecutiveErrors > 0) return "ไม่เสถียร";
  return "เชื่อมต่อแล้ว";
});

const qualityClass = computed(() => ({
  "quality-excellent": status.value.connectionQuality === "excellent",
  "quality-good": status.value.connectionQuality === "good",
  "quality-poor": status.value.connectionQuality === "poor",
  "quality-offline": status.value.connectionQuality === "offline",
}));

const qualityText = computed(() =>
  ({
    excellent: "4G",
    good: "3G",
    poor: "2G",
    offline: "📴",
  })[status.value.connectionQuality] || "",
);

const qualityTextFull = computed(() =>
  ({
    excellent: "🚀 ดีเยี่ยม",
    good: "👍 ดี",
    poor: "⚠️ ช้า",
    offline: "📴 ออฟไลน์",
  })[status.value.connectionQuality] || "ไม่ทราบ",
);

const healthClass = computed(() => ({
  "text-green-400": status.value.health === "excellent",
  "text-yellow-400": status.value.health === "degraded",
  "text-red-400": status.value.health === "critical",
}));

const healthText = computed(() =>
  ({
    excellent: "✅ ดีเยี่ยม",
    degraded: "⚠️ ไม่เสถียร",
    critical: "❌ วิกฤต",
  })[status.value.health] || "ไม่ทราบ",
);

const circuitClass = computed(() => ({
  "text-green-400": status.value.circuitState === "CLOSED",
  "text-yellow-400": status.value.circuitState === "HALF_OPEN",
  "text-red-400": status.value.circuitState === "OPEN",
}));

const successRateClass = computed(() => {
  const rate = parseFloat(metrics.value?.successRate || 100);
  return {
    "text-green-400": rate >= 95,
    "text-yellow-400": rate >= 80 && rate < 95,
    "text-red-400": rate < 80,
  };
});

const toggleDetails = () => {
  showDetails.value = !showDetails.value;
  if (showDetails.value) {
    try {
      metrics.value = getPerformanceMetrics();
    } catch (e) {
      console.warn('Failed to get metrics');
    }
  }
};

const reconnect = async () => {
  await api.reconnect();
};

const clearCache = () => {
  api.clearCache();
  try {
    metrics.value = getPerformanceMetrics();
  } catch (e) {
    console.warn('Failed to get metrics');
  }
};

const exportMetrics = () => {
  if (api.exportMetrics) {
    api.exportMetrics();
  }
};
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
  0%,
  100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.5;
    transform: scale(1.2);
  }
}

.status-text {
  @apply text-gray-300 font-medium;
}

.quality-badge {
  @apply px-1.5 py-0.5 rounded text-xs font-bold;
}

.quality-excellent {
  @apply bg-green-500/20 text-green-400;
}

.quality-good {
  @apply bg-blue-500/20 text-blue-400;
}

.quality-poor {
  @apply bg-yellow-500/20 text-yellow-400;
}

.quality-offline {
  @apply bg-red-500/20 text-red-400;
}

.latency {
  @apply text-gray-500 text-xs;
}

.spinner {
  @apply w-4 h-4 text-yellow-400;
}

.details-panel {
  @apply absolute top-full right-0 mt-2 p-3 rounded-lg z-50;
  @apply bg-gray-900 border border-gray-700 shadow-xl;
  min-width: 280px;
  max-height: 400px;
  overflow-y: auto;
}

.section {
  @apply mb-3 pb-2 border-b border-gray-800;
}

.section:last-child {
  @apply border-0 mb-0 pb-0;
}

.section-title {
  @apply text-xs font-bold text-gray-400 mb-1;
}

.detail-row {
  @apply flex justify-between items-center py-0.5 text-xs;
}

.label {
  @apply text-gray-500;
}

.api-url {
  @apply text-xs text-gray-400 truncate;
  font-family: monospace;
}

.actions {
  @apply mt-3 pt-2 border-t border-gray-700 flex flex-wrap gap-2;
}

.action-btn {
  @apply px-2 py-1 rounded text-xs font-medium;
  @apply bg-gray-700 hover:bg-gray-600 text-gray-300;
  @apply transition-colors;
}

.action-btn:disabled {
  @apply opacity-50 cursor-not-allowed;
}

.action-btn.primary {
  @apply bg-blue-600 hover:bg-blue-500 text-white;
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
