<template>
  <div class="bg-gray-800 rounded-lg p-3 sm:p-6">
    <h2
      class="text-lg sm:text-xl font-bold text-white mb-3 sm:mb-4 flex items-center gap-2"
    >
      <span class="text-xl sm:text-2xl">📜</span>
      Trade History
    </h2>

    <!-- Stats Summary -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4 mb-4 sm:mb-6">
      <div class="bg-gray-700 rounded-lg p-2 sm:p-3 text-center">
        <div class="text-gray-400 text-xs">Total Trades</div>
        <div class="text-base sm:text-xl font-bold text-white">
          {{ stats.total }}
        </div>
      </div>
      <div class="bg-gray-700 rounded-lg p-2 sm:p-3 text-center">
        <div class="text-gray-400 text-xs">Win Rate</div>
        <div class="text-base sm:text-xl font-bold text-blue-400">
          {{ stats.winRate.toFixed(1) }}%
        </div>
      </div>
      <div class="bg-gray-700 rounded-lg p-2 sm:p-3 text-center">
        <div class="text-gray-400 text-xs">Profit Factor</div>
        <div class="text-base sm:text-xl font-bold text-purple-400">
          {{ stats.profitFactor.toFixed(2) }}
        </div>
      </div>
      <div class="bg-gray-700 rounded-lg p-2 sm:p-3 text-center">
        <div class="text-gray-400 text-xs">Total P&L</div>
        <div
          :class="[
            'text-base sm:text-xl font-bold',
            stats.totalPnl >= 0 ? 'text-green-400' : 'text-red-400',
          ]"
        >
          {{ stats.totalPnl >= 0 ? "+" : "" }}{{ stats.totalPnl.toFixed(2) }}
        </div>
      </div>
    </div>

    <!-- History Table (Mobile: Card View) -->
    <div class="hidden sm:block overflow-x-auto">
      <table class="w-full">
        <thead>
          <tr class="text-gray-400 text-xs sm:text-sm border-b border-gray-700">
            <th class="text-left py-2 px-2">Symbol</th>
            <th class="text-left py-2 px-2">Side</th>
            <th class="text-right py-2 px-2">Entry</th>
            <th class="text-right py-2 px-2">Exit</th>
            <th class="text-right py-2 px-2">P&L</th>
            <th class="text-right py-2 px-2 hidden lg:table-cell">Duration</th>
            <th class="text-right py-2 px-2">Date</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="trade in trades"
            :key="trade.id"
            class="border-b border-gray-700/50 hover:bg-gray-700/30 text-xs sm:text-sm"
          >
            <td class="py-2 px-2 text-white font-medium">{{ trade.symbol }}</td>
            <td class="py-2 px-2">
              <span :class="sideClass(trade.side)" class="text-xs">{{
                trade.side
              }}</span>
            </td>
            <td class="py-2 px-2 text-right text-gray-300">
              {{ trade.entry_price.toFixed(2) }}
            </td>
            <td class="py-2 px-2 text-right text-gray-300">
              {{ trade.exit_price?.toFixed(2) || "-" }}
            </td>
            <td class="py-2 px-2 text-right">
              <span :class="pnlClass(trade.pnl)">
                {{ trade.pnl >= 0 ? "+" : "" }}{{ trade.pnl.toFixed(2) }}
              </span>
            </td>
            <td class="py-2 px-2 text-right text-gray-400 hidden lg:table-cell">
              {{ formatDuration(trade) }}
            </td>
            <td class="py-2 px-2 text-right text-gray-400 text-xs">
              {{ formatDate(trade.closed_at) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Mobile Card View -->
    <div class="sm:hidden space-y-2 max-h-64 overflow-y-auto">
      <div
        v-for="trade in trades"
        :key="trade.id"
        class="bg-gray-700 rounded-lg p-3"
      >
        <div class="flex justify-between items-center mb-2">
          <div class="flex items-center gap-2">
            <span class="text-white font-medium text-sm">{{
              trade.symbol
            }}</span>
            <span :class="sideClass(trade.side)" class="text-xs">{{
              trade.side
            }}</span>
          </div>
          <span :class="pnlClass(trade.pnl)" class="font-bold text-sm">
            {{ trade.pnl >= 0 ? "+" : "" }}{{ trade.pnl.toFixed(2) }}
          </span>
        </div>
        <div class="flex justify-between text-xs text-gray-400">
          <span
            >{{ trade.entry_price.toFixed(2) }} →
            {{ trade.exit_price?.toFixed(2) || "-" }}</span
          >
          <span>{{ formatDate(trade.closed_at) }}</span>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="trades.length === 0" class="text-center py-4 sm:py-8">
      <div class="text-3xl sm:text-4xl mb-2">📭</div>
      <div class="text-gray-400 text-sm">No trade history yet</div>
    </div>

    <!-- Load More -->
    <div v-if="trades.length >= 50" class="text-center mt-4">
      <button
        @click="loadMore"
        class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm"
      >
        Load More
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from "vue";
import { useTradingStore } from "@/stores/trading";

const tradingStore = useTradingStore();
const trades = ref([]);

const stats = computed(() => {
  const total = trades.value.length;
  const wins = trades.value.filter((t) => t.pnl > 0).length;
  const losses = trades.value.filter((t) => t.pnl < 0).length;

  const totalWinAmount = trades.value
    .filter((t) => t.pnl > 0)
    .reduce((sum, t) => sum + t.pnl, 0);
  const totalLossAmount = Math.abs(
    trades.value.filter((t) => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0),
  );

  return {
    total,
    wins,
    losses,
    winRate: total > 0 ? (wins / total) * 100 : 0,
    profitFactor:
      totalLossAmount > 0
        ? totalWinAmount / totalLossAmount
        : totalWinAmount > 0
          ? Infinity
          : 0,
    totalPnl: trades.value.reduce((sum, t) => sum + t.pnl, 0),
  };
});

function sideClass(side) {
  return {
    "px-2 py-0.5 rounded text-xs font-bold": true,
    "bg-green-600/30 text-green-400": side === "BUY",
    "bg-red-600/30 text-red-400": side === "SELL",
  };
}

function pnlClass(pnl) {
  return {
    "font-medium": true,
    "text-green-400": pnl >= 0,
    "text-red-400": pnl < 0,
  };
}

function formatDate(dateStr) {
  if (!dateStr) return "-";
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(trade) {
  if (!trade.opened_at || !trade.closed_at) return "-";

  const start = new Date(trade.opened_at);
  const end = new Date(trade.closed_at);
  const diffMs = end - start;

  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

async function fetchHistory() {
  const data = await tradingStore.fetchHistory();
  if (data?.trades) {
    trades.value = data.trades;
  }
}

async function loadMore() {
  // Load more trades with offset
  const data = await tradingStore.fetchHistory(100);
  if (data?.trades) {
    trades.value = data.trades;
  }
}

onMounted(() => {
  fetchHistory();
});
</script>
