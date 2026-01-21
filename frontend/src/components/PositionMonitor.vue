<template>
  <div class="bg-gray-800 rounded-lg p-6">
    <h2 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
      <span class="text-2xl">📊</span>
      Open Positions
    </h2>

    <!-- Empty State -->
    <div v-if="positions.length === 0" class="text-center py-8">
      <div class="text-4xl mb-2">📭</div>
      <div class="text-gray-400">No open positions</div>
    </div>

    <!-- Positions List -->
    <div v-else class="space-y-3">
      <div
        v-for="position in positions"
        :key="position.id"
        class="bg-gray-700 rounded-lg p-4"
      >
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-3">
            <span :class="sideClass(position.side)">
              {{ position.side }}
            </span>
            <span class="text-white font-bold text-lg">{{ position.symbol }}</span>
          </div>
          
          <div :class="pnlClass(position.pnl)">
            {{ position.pnl >= 0 ? '+' : '' }}{{ position.pnl.toFixed(2) }}
            <span class="text-sm">({{ position.pnl_percent.toFixed(2) }}%)</span>
          </div>
        </div>

        <div class="grid grid-cols-4 gap-4 text-sm mb-3">
          <div>
            <div class="text-gray-400">Entry</div>
            <div class="text-white">{{ position.entry_price.toFixed(2) }}</div>
          </div>
          <div>
            <div class="text-gray-400">Current</div>
            <div class="text-white">{{ position.current_price.toFixed(2) }}</div>
          </div>
          <div>
            <div class="text-gray-400">SL</div>
            <div class="text-red-400">{{ position.stop_loss?.toFixed(2) || '-' }}</div>
          </div>
          <div>
            <div class="text-gray-400">TP</div>
            <div class="text-green-400">{{ position.take_profit?.toFixed(2) || '-' }}</div>
          </div>
        </div>

        <!-- Position Progress Bar -->
        <div class="relative h-2 bg-gray-600 rounded-full mb-3">
          <div
            v-if="position.stop_loss && position.take_profit"
            class="absolute h-2 bg-blue-500 rounded-full"
            :style="progressStyle(position)"
          />
          <div
            class="absolute w-2 h-2 bg-white rounded-full transform -translate-x-1/2"
            :style="{ left: currentPricePosition(position) }"
          />
        </div>

        <div class="flex gap-2">
          <button
            @click="showModifyModal(position)"
            class="flex-1 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm"
          >
            ✏️ Modify
          </button>
          <button
            @click="closePosition(position.id)"
            class="flex-1 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm"
          >
            ✕ Close
          </button>
        </div>
      </div>
    </div>

    <!-- Summary -->
    <div v-if="positions.length > 0" class="mt-4 pt-4 border-t border-gray-700">
      <div class="flex justify-between text-sm">
        <span class="text-gray-400">Total Positions: {{ positions.length }}</span>
        <span :class="totalPnlClass">
          Total P&L: {{ totalPnl >= 0 ? '+' : '' }}{{ totalPnl.toFixed(2) }}
        </span>
      </div>
    </div>

    <!-- Modify Modal -->
    <Teleport to="body">
      <div v-if="modifyPosition" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div class="bg-gray-800 rounded-xl p-6 w-full max-w-md">
          <h3 class="text-xl font-bold text-white mb-4">
            Modify {{ modifyPosition.symbol }} {{ modifyPosition.side }}
          </h3>

          <div class="space-y-4">
            <div>
              <label class="text-white block mb-2">Stop Loss</label>
              <input
                type="number"
                v-model.number="modifyData.stop_loss"
                :step="0.01"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              >
            </div>

            <div>
              <label class="text-white block mb-2">Take Profit</label>
              <input
                type="number"
                v-model.number="modifyData.take_profit"
                :step="0.01"
                class="w-full bg-gray-700 text-white rounded-lg px-4 py-2"
              >
            </div>
          </div>

          <div class="flex gap-3 mt-6">
            <button
              @click="saveModify"
              class="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
            >
              Save
            </button>
            <button
              @click="modifyPosition = null"
              class="flex-1 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useTradingStore } from '@/stores/trading'

const tradingStore = useTradingStore()

const modifyPosition = ref(null)
const modifyData = ref({
  stop_loss: null,
  take_profit: null
})

const positions = computed(() => tradingStore.positions)

const totalPnl = computed(() => 
  positions.value.reduce((sum, p) => sum + p.pnl, 0)
)

const totalPnlClass = computed(() => ({
  'font-bold': true,
  'text-green-400': totalPnl.value >= 0,
  'text-red-400': totalPnl.value < 0
}))

function sideClass(side) {
  return {
    'px-2 py-1 rounded text-sm font-bold': true,
    'bg-green-600 text-white': side === 'BUY',
    'bg-red-600 text-white': side === 'SELL'
  }
}

function pnlClass(pnl) {
  return {
    'text-lg font-bold': true,
    'text-green-400': pnl >= 0,
    'text-red-400': pnl < 0
  }
}

function progressStyle(position) {
  // Calculate the width based on SL and TP range
  const range = position.take_profit - position.stop_loss
  const progress = position.current_price - position.stop_loss
  const percent = Math.max(0, Math.min(100, (progress / range) * 100))
  
  return { width: `${percent}%` }
}

function currentPricePosition(position) {
  if (!position.stop_loss || !position.take_profit) return '50%'
  
  const range = position.take_profit - position.stop_loss
  const progress = position.current_price - position.stop_loss
  const percent = Math.max(0, Math.min(100, (progress / range) * 100))
  
  return `${percent}%`
}

function showModifyModal(position) {
  modifyPosition.value = position
  modifyData.value = {
    stop_loss: position.stop_loss,
    take_profit: position.take_profit
  }
}

async function saveModify() {
  await tradingStore.modifyPosition(modifyPosition.value.id, modifyData.value)
  modifyPosition.value = null
}

async function closePosition(positionId) {
  if (confirm('Are you sure you want to close this position?')) {
    await tradingStore.closePosition(positionId)
  }
}

// Fetch positions on mount
tradingStore.fetchPositions()

// Poll positions every 2 seconds
setInterval(() => {
  if (tradingStore.enabled) {
    tradingStore.fetchPositions()
  }
}, 2000)
</script>
