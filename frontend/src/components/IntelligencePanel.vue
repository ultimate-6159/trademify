<template>
  <div class="bg-gray-800 rounded-lg p-6">
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-bold text-white flex items-center gap-2">
        <span class="text-2xl">🧠</span>
        AI Intelligence Layers
        <span class="text-sm font-normal text-gray-400"
          >({{ enabledModules }}/{{ totalModules }} Active)</span
        >
      </h2>

      <div class="flex items-center gap-3">
        <!-- Overall Health -->
        <div :class="overallHealthClass">
          {{ overallHealthEmoji }} {{ overallHealth }}%
        </div>

        <!-- Refresh Button -->
        <button
          @click="refresh"
          :disabled="isLoading"
          class="text-blue-400 hover:text-blue-300"
        >
          <span :class="{ 'animate-spin': isLoading }">🔄</span>
        </button>
      </div>
    </div>

    <!-- Intelligence Stack Visualization -->
    <div class="space-y-2">
      <!-- Titan Core (Top) -->
      <div
        :class="['module-card', getModuleClass('titan')]"
        @click="selectModule('titan')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🏛️</span>
            <div>
              <div class="font-bold">TITAN CORE</div>
              <div class="text-xs text-gray-400">
                Meta-Intelligence Synthesis
              </div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Grade:</span>
              <span :class="getTitanGradeClass(moduleStatus.titan?.grade)">
                {{ moduleStatus.titan?.grade || "N/A" }}
              </span>
            </div>
            <div :class="getStatusBadge(moduleStatus.titan?.active)">
              {{ moduleStatus.titan?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
        <div v-if="moduleStatus.titan?.score" class="mt-2">
          <div class="flex justify-between text-xs text-gray-400 mb-1">
            <span>Titan Score</span>
            <span>{{ moduleStatus.titan.score.toFixed(1) }}%</span>
          </div>
          <div class="w-full bg-gray-700 rounded-full h-2">
            <div
              class="h-2 rounded-full transition-all"
              :class="getScoreBarClass(moduleStatus.titan.score)"
              :style="{ width: moduleStatus.titan.score + '%' }"
            ></div>
          </div>
        </div>
      </div>

      <!-- Omega Brain -->
      <div
        :class="['module-card', getModuleClass('omega')]"
        @click="selectModule('omega')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🧠⚡</span>
            <div>
              <div class="font-bold">OMEGA BRAIN</div>
              <div class="text-xs text-gray-400">
                Institutional-Grade Intelligence
              </div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Grade:</span>
              <span :class="getOmegaGradeClass(moduleStatus.omega?.grade)">
                {{ moduleStatus.omega?.grade || "N/A" }}
              </span>
            </div>
            <div :class="getStatusBadge(moduleStatus.omega?.active)">
              {{ moduleStatus.omega?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Alpha Engine -->
      <div
        :class="['module-card', getModuleClass('alpha')]"
        @click="selectModule('alpha')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🔶</span>
            <div>
              <div class="font-bold">ALPHA ENGINE</div>
              <div class="text-xs text-gray-400">
                Professional Grade Analysis
              </div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Grade:</span>
              <span :class="getAlphaGradeClass(moduleStatus.alpha?.grade)">
                {{ moduleStatus.alpha?.grade || "N/A" }}
              </span>
            </div>
            <div :class="getStatusBadge(moduleStatus.alpha?.active)">
              {{ moduleStatus.alpha?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Quantum Strategy -->
      <div
        :class="['module-card', getModuleClass('quantum')]"
        @click="selectModule('quantum')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">⚛️</span>
            <div>
              <div class="font-bold">QUANTUM STRATEGY</div>
              <div class="text-xs text-gray-400">
                Market Microstructure Analysis
              </div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Grade:</span>
              <span :class="getQuantumGradeClass(moduleStatus.quantum?.grade)">
                {{ moduleStatus.quantum?.grade || "N/A" }}
              </span>
            </div>
            <div :class="getStatusBadge(moduleStatus.quantum?.active)">
              {{ moduleStatus.quantum?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Deep Intelligence -->
      <div
        :class="['module-card', getModuleClass('deep')]"
        @click="selectModule('deep')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🔮</span>
            <div>
              <div class="font-bold">DEEP INTELLIGENCE</div>
              <div class="text-xs text-gray-400">Cross-Asset Correlation</div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Score:</span>
              <span class="text-purple-400">{{
                moduleStatus.deep?.score?.toFixed(0) || "N/A"
              }}</span>
            </div>
            <div :class="getStatusBadge(moduleStatus.deep?.active)">
              {{ moduleStatus.deep?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Neural Brain -->
      <div
        :class="['module-card', getModuleClass('neural')]"
        @click="selectModule('neural')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🧠🔥</span>
            <div>
              <div class="font-bold">NEURAL BRAIN</div>
              <div class="text-xs text-gray-400">Pattern DNA Analysis</div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">DNA:</span>
              <span class="text-orange-400">{{
                moduleStatus.neural?.dna_score?.toFixed(0) || "N/A"
              }}</span>
            </div>
            <div :class="getStatusBadge(moduleStatus.neural?.active)">
              {{ moduleStatus.neural?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Continuous Learning -->
      <div
        :class="['module-card', getModuleClass('learning')]"
        @click="selectModule('learning')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">📚</span>
            <div>
              <div class="font-bold">CONTINUOUS LEARNING</div>
              <div class="text-xs text-gray-400">Market Adaptation System</div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Cycles:</span>
              <span class="text-blue-400">{{
                moduleStatus.learning?.cycles || 0
              }}</span>
            </div>
            <div :class="getStatusBadge(moduleStatus.learning?.active)">
              {{ moduleStatus.learning?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Advanced Intelligence -->
      <div
        :class="['module-card', getModuleClass('advanced')]"
        @click="selectModule('advanced')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🧠</span>
            <div>
              <div class="font-bold">ADVANCED INTELLIGENCE</div>
              <div class="text-xs text-gray-400">Multi-Timeframe Analysis</div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Regime:</span>
              <span :class="getRegimeClass(moduleStatus.advanced?.regime)">
                {{ moduleStatus.advanced?.regime || "N/A" }}
              </span>
            </div>
            <div :class="getStatusBadge(moduleStatus.advanced?.active)">
              {{ moduleStatus.advanced?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Smart Brain -->
      <div
        :class="['module-card', getModuleClass('smart')]"
        @click="selectModule('smart')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">💡</span>
            <div>
              <div class="font-bold">SMART BRAIN</div>
              <div class="text-xs text-gray-400">Trade Journal & Memory</div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Patterns:</span>
              <span class="text-yellow-400">{{
                moduleStatus.smart?.patterns || 0
              }}</span>
            </div>
            <div :class="getStatusBadge(moduleStatus.smart?.active)">
              {{ moduleStatus.smart?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Pro Trading Features -->
      <div
        :class="['module-card', getModuleClass('pro')]"
        @click="selectModule('pro')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">⭐</span>
            <div>
              <div class="font-bold">PRO TRADING FEATURES</div>
              <div class="text-xs text-gray-400">
                Sessions / News / Correlation
              </div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Session:</span>
              <span :class="getSessionClass(moduleStatus.pro?.session)">
                {{ moduleStatus.pro?.session || "N/A" }}
              </span>
            </div>
            <div :class="getStatusBadge(moduleStatus.pro?.active)">
              {{ moduleStatus.pro?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>

      <!-- Risk Guardian (Base) -->
      <div
        :class="['module-card', getModuleClass('risk')]"
        @click="selectModule('risk')"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🛡️</span>
            <div>
              <div class="font-bold">RISK GUARDIAN</div>
              <div class="text-xs text-gray-400">Account Protection</div>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-sm">
              <span class="text-gray-400">Level:</span>
              <span :class="getRiskLevelClass(moduleStatus.risk?.level)">
                {{ moduleStatus.risk?.level || "N/A" }}
              </span>
            </div>
            <div :class="getStatusBadge(moduleStatus.risk?.active)">
              {{ moduleStatus.risk?.active ? "✓" : "○" }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Module Details Panel -->
    <div v-if="selectedModuleName" class="mt-6 bg-gray-700 rounded-lg p-4">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-bold text-white">
          {{ getModuleTitle(selectedModuleName) }} Details
        </h3>
        <button
          @click="selectedModuleName = null"
          class="text-gray-400 hover:text-white"
        >
          ✕
        </button>
      </div>

      <ModuleDetails
        :module-name="selectedModuleName"
        :data="moduleStatus[selectedModuleName]"
      />
    </div>

    <!-- Legend -->
    <div class="mt-6 flex flex-wrap gap-4 text-xs text-gray-400">
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full bg-green-500"></span>
        <span>Active & Healthy</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full bg-yellow-500"></span>
        <span>Warning</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full bg-red-500"></span>
        <span>Error/Blocked</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full bg-gray-500"></span>
        <span>Inactive</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import ModuleDetails from "./ModuleDetails.vue";
import api from "../services/api";

const props = defineProps({
  autoRefresh: {
    type: Boolean,
    default: true,
  },
});

const emit = defineEmits(["module-selected"]);

const isLoading = ref(false);
const selectedModuleName = ref(null);
const moduleStatus = ref({
  titan: { active: true, grade: "🏛️ TITAN SUPREME", score: 92.5 },
  omega: { active: true, grade: "Ω+", score: 88.0 },
  alpha: { active: true, grade: "A+", score: 85.0 },
  quantum: { active: true, grade: "QUANTUM", score: 78.0 },
  deep: { active: true, score: 75.0 },
  neural: { active: true, dna_score: 82.0 },
  learning: { active: true, cycles: 150 },
  advanced: { active: true, regime: "UPTREND" },
  smart: { active: true, patterns: 1250 },
  pro: { active: true, session: "LONDON" },
  risk: { active: true, level: "SAFE" },
});

let refreshInterval = null;

const totalModules = computed(() => Object.keys(moduleStatus.value).length);

const enabledModules = computed(() => {
  return Object.values(moduleStatus.value).filter((m) => m.active).length;
});

const overallHealth = computed(() => {
  const scores = Object.values(moduleStatus.value)
    .filter(
      (m) => m.active && (m.score !== undefined || m.dna_score !== undefined),
    )
    .map((m) => m.score || m.dna_score || 50);

  if (scores.length === 0) return 0;
  return Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
});

const overallHealthEmoji = computed(() => {
  const health = overallHealth.value;
  if (health >= 90) return "💚";
  if (health >= 75) return "💛";
  if (health >= 50) return "🧡";
  return "❤️";
});

const overallHealthClass = computed(() => {
  const health = overallHealth.value;
  if (health >= 90) return "text-green-400 font-bold";
  if (health >= 75) return "text-yellow-400 font-bold";
  if (health >= 50) return "text-orange-400 font-bold";
  return "text-red-400 font-bold";
});

const refresh = async () => {
  isLoading.value = true;
  try {
    const status = await api.getIntelligenceStatus();
    if (status && !status._isMock) {
      moduleStatus.value = { ...moduleStatus.value, ...status };
    }
  } catch (e) {
    console.error("Failed to refresh intelligence status:", e);
  } finally {
    isLoading.value = false;
  }
};

const selectModule = (name) => {
  selectedModuleName.value = selectedModuleName.value === name ? null : name;
  emit("module-selected", selectedModuleName.value);
};

const getModuleClass = (name) => {
  const mod = moduleStatus.value[name];
  if (!mod?.active) return "opacity-50 border-gray-600";
  if (mod.error) return "border-red-500 bg-red-900/20";
  if (mod.warning) return "border-yellow-500 bg-yellow-900/20";
  return "border-green-500/30 bg-green-900/10 hover:border-green-400";
};

const getStatusBadge = (active) => {
  return active
    ? "w-8 h-8 rounded-full bg-green-500/20 text-green-400 flex items-center justify-center font-bold"
    : "w-8 h-8 rounded-full bg-gray-600/20 text-gray-500 flex items-center justify-center";
};

const getScoreBarClass = (score) => {
  if (score >= 85) return "bg-green-500";
  if (score >= 70) return "bg-yellow-500";
  if (score >= 50) return "bg-orange-500";
  return "bg-red-500";
};

// Grade classes for each module type
const getTitanGradeClass = (grade) => {
  if (!grade) return "text-gray-400";
  if (grade.includes("SUPREME")) return "text-yellow-300 font-bold";
  if (grade.includes("ELITE")) return "text-purple-400 font-bold";
  if (grade.includes("PRIME")) return "text-blue-400 font-bold";
  if (grade.includes("CORE")) return "text-cyan-400";
  if (grade.includes("BASE")) return "text-gray-400";
  return "text-red-400";
};

const getOmegaGradeClass = (grade) => {
  if (!grade) return "text-gray-400";
  if (grade === "Ω+") return "text-yellow-300 font-bold";
  if (grade === "Ω") return "text-purple-400 font-bold";
  if (grade === "α+") return "text-blue-400 font-bold";
  if (grade === "α") return "text-cyan-400";
  if (grade === "β") return "text-green-400";
  if (grade === "γ") return "text-gray-400";
  return "text-red-400";
};

const getAlphaGradeClass = (grade) => {
  if (!grade) return "text-gray-400";
  if (grade.includes("A+") || grade.includes("ELITE"))
    return "text-yellow-300 font-bold";
  if (grade.includes("A") || grade.includes("PRIME"))
    return "text-green-400 font-bold";
  if (grade.includes("B")) return "text-blue-400";
  if (grade.includes("C")) return "text-yellow-400";
  return "text-red-400";
};

const getQuantumGradeClass = (grade) => {
  if (!grade) return "text-gray-400";
  if (grade === "QUANTUM") return "text-purple-400 font-bold animate-pulse";
  if (grade === "STELLAR") return "text-blue-400 font-bold";
  return "text-gray-400";
};

const getRegimeClass = (regime) => {
  if (!regime) return "text-gray-400";
  if (regime === "UPTREND") return "text-green-400";
  if (regime === "DOWNTREND") return "text-red-400";
  if (regime === "RANGING") return "text-yellow-400";
  return "text-gray-400";
};

const getSessionClass = (session) => {
  if (!session) return "text-gray-400";
  if (session === "LONDON") return "text-blue-400";
  if (session === "NEW_YORK") return "text-green-400";
  if (session === "TOKYO") return "text-red-400";
  if (session === "SYDNEY") return "text-purple-400";
  return "text-gray-400";
};

const getRiskLevelClass = (level) => {
  if (!level) return "text-gray-400";
  if (level === "SAFE") return "text-green-400";
  if (level === "CAUTION") return "text-yellow-400";
  if (level === "WARNING") return "text-orange-400";
  if (level === "DANGER" || level === "CRITICAL") return "text-red-400";
  return "text-gray-400";
};

const getModuleTitle = (name) => {
  const titles = {
    titan: "🏛️ TITAN CORE",
    omega: "🧠⚡ OMEGA BRAIN",
    alpha: "🔶 ALPHA ENGINE",
    quantum: "⚛️ QUANTUM STRATEGY",
    deep: "🔮 DEEP INTELLIGENCE",
    neural: "🧠🔥 NEURAL BRAIN",
    learning: "📚 CONTINUOUS LEARNING",
    advanced: "🧠 ADVANCED INTELLIGENCE",
    smart: "💡 SMART BRAIN",
    pro: "⭐ PRO TRADING",
    risk: "🛡️ RISK GUARDIAN",
  };
  return titles[name] || name.toUpperCase();
};

onMounted(() => {
  refresh();
  if (props.autoRefresh) {
    refreshInterval = setInterval(refresh, 10000); // Refresh every 10 seconds
  }
});

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
});
</script>

<style scoped>
.module-card {
  @apply p-4 rounded-lg border cursor-pointer transition-all;
}

.module-card:hover {
  @apply transform scale-[1.01];
}
</style>
