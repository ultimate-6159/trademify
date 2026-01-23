<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white flex items-center gap-3">
          <span class="text-4xl">🔮</span>
          AI Intelligence Center
        </h1>
        <p class="text-gray-400 mt-1">
          20-Layer Omniscient Analysis System • {{ activeLayersCount }}/20 Layers Active
        </p>
      </div>

      <div class="flex items-center gap-4">
        <!-- Symbol Selector -->
        <select
          v-model="selectedSymbol"
          class="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
        >
          <option v-for="symbol in symbols" :key="symbol" :value="symbol">
            {{ symbol }}
          </option>
        </select>

        <!-- Refresh Button -->
        <button
          @click="refreshAll"
          :disabled="isLoading"
          class="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-semibold transition-colors flex items-center gap-2"
        >
          <span :class="{ 'animate-spin': isLoading }">🔄</span>
          Refresh
        </button>
      </div>
    </div>

    <!-- Ultra Stats Row -->
    <div class="grid grid-cols-5 gap-4">
      <div class="bg-gradient-to-br from-purple-900/50 to-purple-700/30 rounded-lg p-4 text-center border border-purple-500/50">
        <div class="text-purple-300 text-xs mb-1">🏛️ TITAN SCORE</div>
        <div class="text-3xl font-bold text-purple-400">{{ titanScore }}</div>
      </div>
      <div class="bg-gradient-to-br from-indigo-900/50 to-indigo-700/30 rounded-lg p-4 text-center border border-indigo-500/50">
        <div class="text-indigo-300 text-xs mb-1">⚡ ULTRA 10x</div>
        <div class="text-3xl font-bold text-indigo-400">{{ ultraScore }}</div>
      </div>
      <div class="bg-gradient-to-br from-yellow-900/50 to-yellow-700/30 rounded-lg p-4 text-center border border-yellow-500/50">
        <div class="text-yellow-300 text-xs mb-1">👑 SUPREME 20x</div>
        <div class="text-3xl font-bold text-yellow-400">{{ supremeScore }}</div>
      </div>
      <div class="bg-gradient-to-br from-cyan-900/50 to-cyan-700/30 rounded-lg p-4 text-center border border-cyan-500/50">
        <div class="text-cyan-300 text-xs mb-1">🌌 TRANSCEND 50x</div>
        <div class="text-3xl font-bold text-cyan-400">{{ transcendentScore }}</div>
      </div>
      <div class="bg-gradient-to-br from-pink-900/50 to-pink-700/30 rounded-lg p-4 text-center border border-pink-500/50">
        <div class="text-pink-300 text-xs mb-1">🔮 OMNISCIENT 100x</div>
        <div class="text-3xl font-bold text-pink-400">{{ omniscientScore }}</div>
      </div>
    </div>

    <!-- Signal Quality Filter -->
    <div class="bg-gray-800 rounded-lg p-4 border border-blue-500/30">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <span class="text-xl">🎚️</span>
          <div>
            <div class="text-white font-medium">Signal Quality Filter</div>
            <div class="text-gray-400 text-sm">Current: {{ qualityFilter.current }}</div>
          </div>
        </div>
        <div class="flex gap-2">
          <span 
            v-for="(level, key) in qualityFilter.levels" 
            :key="key"
            :class="[
              'px-3 py-1 rounded-full text-xs font-medium',
              qualityFilter.current === key 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-700 text-gray-400'
            ]"
          >
            {{ key }} ({{ level.threshold }}%+)
          </span>
        </div>
      </div>
    </div>

    <!-- 20 Intelligence Layers Grid -->
    <div class="space-y-6">
      <!-- Foundation Layers (1-10) -->
      <div>
        <h3 class="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <span class="text-blue-400">🔧</span> Foundation Layers (1-10)
        </h3>
        <div class="grid grid-cols-5 gap-3">
          <div 
            v-for="layer in foundationLayers" 
            :key="layer.id"
            :class="[
              'rounded-lg p-3 border transition-all cursor-pointer hover:scale-105',
              layer.active 
                ? 'bg-gray-800 border-green-500/50' 
                : 'bg-gray-900 border-gray-700 opacity-60'
            ]"
          >
            <div class="flex items-center gap-2 mb-2">
              <span class="text-xl">{{ layer.icon }}</span>
              <div 
                :class="[
                  'w-2 h-2 rounded-full',
                  layer.active ? 'bg-green-400 animate-pulse' : 'bg-gray-600'
                ]"
              ></div>
            </div>
            <div class="text-white font-medium text-sm">{{ layer.name }}</div>
            <div class="text-gray-400 text-xs mt-1 line-clamp-2">{{ layer.description }}</div>
          </div>
        </div>
      </div>

      <!-- Advanced Layers (11-16) -->
      <div>
        <h3 class="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <span class="text-purple-400">🧠</span> Advanced Layers (11-16)
        </h3>
        <div class="grid grid-cols-6 gap-3">
          <div 
            v-for="layer in advancedLayers" 
            :key="layer.id"
            :class="[
              'rounded-lg p-3 border transition-all cursor-pointer hover:scale-105',
              layer.active 
                ? 'bg-gray-800 border-purple-500/50' 
                : 'bg-gray-900 border-gray-700 opacity-60'
            ]"
          >
            <div class="flex items-center gap-2 mb-2">
              <span class="text-xl">{{ layer.icon }}</span>
              <div 
                :class="[
                  'w-2 h-2 rounded-full',
                  layer.active ? 'bg-purple-400 animate-pulse' : 'bg-gray-600'
                ]"
              ></div>
            </div>
            <div class="text-white font-medium text-sm">{{ layer.name }}</div>
            <div class="text-gray-400 text-xs mt-1 line-clamp-2">{{ layer.description }}</div>
          </div>
        </div>
      </div>

      <!-- Ultra Intelligence Layers (17-20) -->
      <div>
        <h3 class="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <span class="text-yellow-400">⚡</span> Ultra Intelligence Layers (17-20)
          <span class="text-xs bg-gradient-to-r from-yellow-500 to-pink-500 text-white px-2 py-1 rounded-full">
            10x → 100x SMARTER
          </span>
        </h3>
        <div class="grid grid-cols-4 gap-4">
          <div 
            v-for="layer in ultraLayers" 
            :key="layer.id"
            :class="[
              'rounded-lg p-4 border-2 transition-all cursor-pointer hover:scale-105',
              layer.active 
                ? getUltraLayerClass(layer.id) 
                : 'bg-gray-900 border-gray-700 opacity-60'
            ]"
          >
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center gap-2">
                <span class="text-2xl">{{ layer.icon }}</span>
                <span 
                  :class="[
                    'px-2 py-0.5 rounded text-xs font-bold',
                    layer.active ? 'bg-white/20 text-white' : 'bg-gray-700 text-gray-400'
                  ]"
                >
                  {{ layer.multiplier }}
                </span>
              </div>
              <div 
                :class="[
                  'w-3 h-3 rounded-full',
                  layer.active ? 'bg-green-400 animate-pulse' : 'bg-gray-600'
                ]"
              ></div>
            </div>
            <div class="text-white font-bold">{{ layer.name }}</div>
            <div class="text-gray-300 text-xs mt-1">{{ layer.description }}</div>
            <div class="mt-3 flex flex-wrap gap-1">
              <span 
                v-for="feature in layer.features?.slice(0, 4)" 
                :key="feature"
                class="text-xs px-2 py-0.5 rounded bg-white/10 text-gray-300"
              >
                {{ feature }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Last Decisions from Ultra Layers -->
    <div v-if="hasUltraDecisions" class="bg-gray-800 rounded-lg p-6 border border-yellow-500/30">
      <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
        <span>🎯</span> Latest Ultra Intelligence Decisions
      </h3>
      
      <div class="grid grid-cols-2 gap-4">
        <!-- Transcendent Decision -->
        <div v-if="lastDecisions.transcendent" class="bg-gray-700/50 rounded-lg p-4">
          <div class="flex items-center gap-2 mb-3">
            <span class="text-xl">🌌</span>
            <span class="text-cyan-400 font-bold">Transcendent Intelligence (50x)</span>
          </div>
          <div class="grid grid-cols-2 gap-2 text-sm">
            <div class="text-gray-400">Quantum State:</div>
            <div class="text-white">{{ lastDecisions.transcendent.quantum_state || 'N/A' }}</div>
            <div class="text-gray-400">Win Probability:</div>
            <div class="text-green-400">{{ formatPercent(lastDecisions.transcendent.win_probability) }}%</div>
            <div class="text-gray-400">Score:</div>
            <div class="text-cyan-400">{{ lastDecisions.transcendent.transcendent_score?.toFixed(0) || 0 }}/100</div>
            <div class="text-gray-400">Can Trade:</div>
            <div :class="lastDecisions.transcendent.can_trade ? 'text-green-400' : 'text-red-400'">
              {{ lastDecisions.transcendent.can_trade ? 'YES ✓' : 'NO ✗' }}
            </div>
          </div>
        </div>

        <!-- Omniscient Decision -->
        <div v-if="lastDecisions.omniscient" class="bg-gray-700/50 rounded-lg p-4">
          <div class="flex items-center gap-2 mb-3">
            <span class="text-xl">🔮</span>
            <span class="text-pink-400 font-bold">Omniscient Intelligence (100x)</span>
          </div>
          <div class="grid grid-cols-2 gap-2 text-sm">
            <div class="text-gray-400">Consciousness:</div>
            <div class="text-white">{{ lastDecisions.omniscient.consciousness_level || 'N/A' }}</div>
            <div class="text-gray-400">Edge:</div>
            <div class="text-green-400">{{ lastDecisions.omniscient.edge?.toFixed(2) || 0 }}%</div>
            <div class="text-gray-400">Score:</div>
            <div class="text-pink-400">{{ lastDecisions.omniscient.omniscient_score?.toFixed(0) || 0 }}/100</div>
            <div class="text-gray-400">Physics:</div>
            <div class="text-blue-400">{{ lastDecisions.omniscient.physics_state || 'N/A' }}</div>
          </div>
          <!-- Prophecies -->
          <div v-if="lastDecisions.omniscient.prophecies?.length" class="mt-3 border-t border-gray-600 pt-2">
            <div class="text-gray-400 text-xs mb-1">🔮 Prophecies:</div>
            <div v-for="prophecy in lastDecisions.omniscient.prophecies.slice(0, 2)" :key="prophecy" class="text-xs text-yellow-300">
              • {{ prophecy }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Layer Features Detail -->
    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
        <span>📚</span> All 20 Layers Feature Summary
      </h3>
      
      <div class="grid grid-cols-4 gap-3 max-h-96 overflow-y-auto">
        <div 
          v-for="layer in allLayers" 
          :key="layer.id"
          class="bg-gray-700/50 rounded-lg p-3"
        >
          <div class="flex items-center gap-2 mb-2">
            <span>{{ layer.icon }}</span>
            <span class="text-white text-sm font-medium">{{ layer.id }}. {{ layer.name }}</span>
            <span 
              v-if="layer.multiplier"
              class="text-xs px-1 py-0.5 rounded bg-yellow-500/20 text-yellow-400"
            >
              {{ layer.multiplier }}
            </span>
          </div>
          <div class="flex flex-wrap gap-1">
            <span 
              v-for="feature in layer.features" 
              :key="feature"
              class="text-xs px-1.5 py-0.5 rounded bg-gray-600 text-gray-300"
            >
              {{ feature }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Auto-Refresh Status -->
    <div class="text-center text-gray-500 text-sm">
      🔄 Auto-refresh every 30 seconds • Last updated: {{ lastUpdated }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import api from "../services/api";

// State
const isLoading = ref(false);
const selectedSymbol = ref("EURUSDm");
const lastUpdated = ref(new Date().toLocaleTimeString());
let refreshInterval = null;
const symbols = ref(["EURUSDm", "GBPUSDm", "XAUUSDm"]);

// Layer data from API
const allLayers = ref([]);
const lastDecisions = ref({});
const qualityFilter = ref({
  current: "MEDIUM",
  levels: {
    PREMIUM: { threshold: 85 },
    HIGH: { threshold: 75 },
    MEDIUM: { threshold: 65 },
    LOW: { threshold: 50 },
  }
});

// Computed
const foundationLayers = computed(() => allLayers.value.filter(l => l.id >= 1 && l.id <= 10));
const advancedLayers = computed(() => allLayers.value.filter(l => l.id >= 11 && l.id <= 16));
const ultraLayers = computed(() => allLayers.value.filter(l => l.id >= 17 && l.id <= 20));

const activeLayersCount = computed(() => allLayers.value.filter(l => l.active).length);

const titanScore = computed(() => {
  const layer = allLayers.value.find(l => l.id === 16);
  return layer?.active ? '✓' : '–';
});

const ultraScore = computed(() => {
  const decision = lastDecisions.value.ultra;
  if (decision?.confidence) return decision.confidence.toFixed(0) + '%';
  const layer = allLayers.value.find(l => l.id === 17);
  return layer?.active ? '✓' : '–';
});

const supremeScore = computed(() => {
  const decision = lastDecisions.value.supreme;
  if (decision?.supreme_score) return decision.supreme_score.toFixed(0) + '%';
  const layer = allLayers.value.find(l => l.id === 18);
  return layer?.active ? '✓' : '–';
});

const transcendentScore = computed(() => {
  const decision = lastDecisions.value.transcendent;
  if (decision?.transcendent_score) return decision.transcendent_score.toFixed(0) + '%';
  const layer = allLayers.value.find(l => l.id === 19);
  return layer?.active ? '✓' : '–';
});

const omniscientScore = computed(() => {
  const decision = lastDecisions.value.omniscient;
  if (decision?.omniscient_score) return decision.omniscient_score.toFixed(0) + '%';
  const layer = allLayers.value.find(l => l.id === 20);
  return layer?.active ? '✓' : '–';
});

const hasUltraDecisions = computed(() => {
  return lastDecisions.value.transcendent || lastDecisions.value.omniscient;
});

// Methods
const formatPercent = (val) => {
  if (!val && val !== 0) return 0;
  return (val * 100).toFixed(0);
};

const getUltraLayerClass = (id) => {
  switch(id) {
    case 17: return 'bg-gradient-to-br from-indigo-900/50 to-indigo-700/30 border-indigo-500';
    case 18: return 'bg-gradient-to-br from-yellow-900/50 to-yellow-700/30 border-yellow-500';
    case 19: return 'bg-gradient-to-br from-cyan-900/50 to-cyan-700/30 border-cyan-500';
    case 20: return 'bg-gradient-to-br from-pink-900/50 to-pink-700/30 border-pink-500';
    default: return 'bg-gray-800 border-gray-600';
  }
};

const refreshAll = async () => {
  isLoading.value = true;
  try {
    // Fetch all layers data from new API endpoint
    const response = await api.get('/api/v1/intelligence/layers');
    
    if (response && response.layers) {
      allLayers.value = response.layers;
      lastDecisions.value = response.last_decisions || {};
      qualityFilter.value = response.quality_filter || qualityFilter.value;
    }
    
    lastUpdated.value = new Date().toLocaleTimeString();
    console.log("Layers loaded:", allLayers.value.length, "Active:", activeLayersCount.value);
  } catch (e) {
    console.error("Failed to refresh layers:", e);
    // Load default layers if API fails
    loadDefaultLayers();
  } finally {
    isLoading.value = false;
  }
};

const loadDefaultLayers = () => {
  allLayers.value = [
    { id: 1, name: "Pattern Recognition", icon: "🎯", description: "FAISS Pattern Matching", features: ["FAISS Index", "Z-Score", "Correlation"], active: false },
    { id: 2, name: "Technical Analysis", icon: "📊", description: "RSI, MACD, BB, EMA", features: ["RSI", "MACD", "BB", "EMA"], active: false },
    { id: 3, name: "Volume Analysis", icon: "📈", description: "Volume Profile, OBV", features: ["Volume Ratio", "OBV", "Spike"], active: false },
    { id: 4, name: "Multi-Timeframe", icon: "🔄", description: "H1/H4/D1 Confluence", features: ["HTF Trend", "Confluence"], active: false },
    { id: 5, name: "Market Regime", icon: "🌊", description: "Trend/Range Detection", features: ["Trend", "Range", "Volatile"], active: false },
    { id: 6, name: "Risk Guardian", icon: "🛡️", description: "2% Risk, 5% Daily Loss", features: ["Position Size", "Daily Limit"], active: false },
    { id: 7, name: "Pro Features", icon: "🏆", description: "Session, News, Trailing", features: ["Session", "News", "Trailing"], active: false },
    { id: 8, name: "Smart Brain", icon: "🧠", description: "Trade Journal, Pattern Memory", features: ["Journal", "Memory", "Adaptive"], active: false },
    { id: 9, name: "Advanced Intelligence", icon: "🎓", description: "Kelly Criterion, S/R", features: ["Kelly", "Regime", "S/R"], active: false },
    { id: 10, name: "Continuous Learning", icon: "📚", description: "Online Learning, Cycle", features: ["Online", "Cycle", "Optimize"], active: false },
    { id: 11, name: "Neural Brain", icon: "🧬", description: "Pattern DNA, State Machine", features: ["DNA", "State", "Anomaly"], active: false },
    { id: 12, name: "Deep Intelligence", icon: "🔮", description: "Cross-Asset, Predictive", features: ["Cross-Asset", "Predictive"], active: false },
    { id: 13, name: "Quantum Strategy", icon: "⚛️", description: "Microstructure, Fractal", features: ["Smart Money", "GARCH", "Hurst"], active: false },
    { id: 14, name: "Alpha Engine", icon: "🏅", description: "Order Flow, Liquidity", features: ["Order Flow", "POC", "Divergence"], active: false },
    { id: 15, name: "Omega Brain", icon: "Ω", description: "Institutional, Manipulation", features: ["Big Money", "Stop Hunt"], active: false },
    { id: 16, name: "Titan Core", icon: "🏛️", description: "Meta-Intelligence", features: ["Consensus", "Ensemble", "Self-Improve"], active: false },
    { id: 17, name: "Ultra Intelligence", icon: "⚡", description: "10x Smarter - SMC", features: ["SMC", "Session", "Volatility", "Liquidity"], multiplier: "10x", active: false },
    { id: 18, name: "Supreme Intelligence", icon: "👑", description: "20x Smarter - Hedge Fund", features: ["Order Flow", "Entropy", "Fractal", "Win Prob"], multiplier: "20x", active: false },
    { id: 19, name: "Transcendent Intelligence", icon: "🌌", description: "50x Smarter - Quantum", features: ["Quantum", "7D Analysis", "Black Swan", "Purity"], multiplier: "50x", active: false },
    { id: 20, name: "Omniscient Intelligence", icon: "🔮", description: "100x Smarter - All-Knowing", features: ["Physics", "Neural", "Chaos", "Game Theory", "Prophecy"], multiplier: "100x", active: false },
  ];
};

// Watch for symbol changes
watch(selectedSymbol, () => {
  refreshAll();
});

onMounted(() => {
  refreshAll();
  // Auto-refresh every 30 seconds
  refreshInterval = setInterval(() => {
    refreshAll();
  }, 30000);
});

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
});
</script>
