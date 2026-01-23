<template>
  <div class="space-y-4 sm:space-y-6 pb-20 md:pb-0">
    <!-- Header Section -->
    <div
      class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-0"
    >
      <div>
        <h2 class="text-xl sm:text-3xl font-bold text-white">Dashboard</h2>
        <p class="text-gray-400 text-sm sm:text-base mt-1">
          AI Pattern Recognition
        </p>
      </div>

      <div class="flex flex-wrap items-center gap-2 sm:gap-4">
        <!-- Symbol Selector -->
        <select
          v-model="selectedSymbol"
          class="bg-dark-100 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary text-sm flex-1 sm:flex-none"
        >
          <option v-for="symbol in symbols" :key="symbol" :value="symbol">
            {{ symbol }}
          </option>
        </select>

        <!-- Timeframe Selector -->
        <select
          v-model="selectedTimeframe"
          class="bg-dark-100 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary text-sm"
        >
          <option value="M5">M5</option>
          <option value="M15">M15</option>
          <option value="H1">H1</option>
        </select>

        <!-- Generate Signal Button -->
        <button
          @click="generateSignal"
          :disabled="isLoading"
          class="btn btn-primary flex items-center space-x-1 sm:space-x-2 text-sm px-3 py-2 flex-1 sm:flex-none justify-center"
        >
          <svg
            v-if="isLoading"
            class="animate-spin h-4 w-4 sm:h-5 sm:w-5"
            viewBox="0 0 24 24"
          >
            <circle
              class="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              stroke-width="4"
              fill="none"
            />
            <path
              class="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <span>{{ isLoading ? "Analyzing..." : "Analyze" }}</span>
        </button>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
      <!-- Signal Box -->
      <div class="lg:col-span-1 order-2 lg:order-1">
        <SignalBox :signal="currentSignal" />
      </div>

      <!-- Chart -->
      <div class="lg:col-span-2 order-1 lg:order-2">
        <div class="card h-64 sm:h-96 p-3 sm:p-4">
          <h3
            class="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4"
          >
            Pattern Chart
          </h3>
          <PatternChart
            :current-pattern="currentPattern"
            :matched-patterns="matchedPatterns"
            :projected-movement="projectedMovement"
          />
        </div>
      </div>
    </div>

    <!-- Details Section -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
      <!-- Vote Details -->
      <div class="card p-3 sm:p-4">
        <h3 class="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">
          Vote Analysis
        </h3>
        <VoteChart :vote-details="voteDetails" />
      </div>

      <!-- Matched Patterns List -->
      <div class="card p-3 sm:p-4">
        <h3 class="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">
          Matched Patterns
        </h3>
        <PatternList :patterns="matchedPatternsList" />
      </div>
    </div>

    <!-- Price Projection -->
    <div class="card p-3 sm:p-4" v-if="currentSignal?.price_projection">
      <h3 class="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">
        Price Projection
      </h3>
      <PriceProjection :projection="currentSignal.price_projection" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from "vue";
import { useSignalStore } from "../stores/signal";
import SignalBox from "../components/SignalBox.vue";
import PatternChart from "../components/PatternChart.vue";
import VoteChart from "../components/VoteChart.vue";
import PatternList from "../components/PatternList.vue";
import PriceProjection from "../components/PriceProjection.vue";

const signalStore = useSignalStore();

const symbols = ref([
  "EURUSD",
  "GBPUSD",
  "USDJPY",
  "BTCUSDT",
  "ETHUSDT",
  "XAUUSD",
]);
const selectedSymbol = ref("EURUSD");
const selectedTimeframe = ref("H1");

const isLoading = computed(() => signalStore.isLoading);
const currentSignal = computed(() => signalStore.currentSignal);

const currentPattern = computed(() => {
  // Generate sample current pattern
  const length = 60;
  const pattern = [];
  let price = 100;
  for (let i = 0; i < length; i++) {
    price += (Math.random() - 0.5) * 2;
    pattern.push(price);
  }
  return pattern;
});

const matchedPatterns = computed(() => {
  if (!currentSignal.value?.matched_patterns) return [];
  // Generate sample matched patterns for visualization
  return currentSignal.value.matched_patterns.slice(0, 5).map((p, idx) => {
    const pattern = [];
    let price = 100 + (Math.random() - 0.5) * 5;
    for (let i = 0; i < 60; i++) {
      price += (Math.random() - 0.5) * 2;
      pattern.push(price);
    }
    return {
      ...p,
      data: pattern,
    };
  });
});

const projectedMovement = computed(() => {
  return currentSignal.value?.average_movement || [];
});

const voteDetails = computed(() => {
  return (
    currentSignal.value?.vote_details || { bullish: 0, bearish: 0, total: 0 }
  );
});

const matchedPatternsList = computed(() => {
  return currentSignal.value?.matched_patterns || [];
});

async function generateSignal() {
  signalStore.setSymbol(selectedSymbol.value);
  signalStore.setTimeframe(selectedTimeframe.value);

  try {
    await signalStore.fetchSampleSignal();
  } catch (error) {
    console.error("Failed to generate signal:", error);
  }
}

onMounted(() => {
  // Generate initial signal
  generateSignal();
});
</script>
