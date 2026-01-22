<template>
  <div class="min-h-screen bg-gray-900 text-white p-4">
    <!-- Header -->
    <div class="flex items-center justify-between mb-4">
      <div>
        <h1 class="text-3xl font-bold flex items-center gap-3">
          <span class="text-4xl">🔬</span>
          AI Trading Control Center
        </h1>
        <p class="text-gray-400 mt-1">
          Real-time 16-Layer Analysis → Signal → Execution
        </p>
      </div>

      <div class="flex items-center gap-4">
        <!-- Tab Switcher -->
        <div class="flex bg-gray-800 rounded-lg p-1">
          <button
            @click="activeTab = 'pipeline'"
            class="px-4 py-2 rounded-md text-sm font-semibold transition-all"
            :class="
              activeTab === 'pipeline'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white'
            "
          >
            🔬 Pipeline
          </button>
          <button
            @click="activeTab = 'health'"
            class="px-4 py-2 rounded-md text-sm font-semibold transition-all"
            :class="
              activeTab === 'health'
                ? 'bg-green-600 text-white'
                : 'text-gray-400 hover:text-white'
            "
          >
            🏥 System Health
          </button>
        </div>

        <!-- Symbol Selector (only in Pipeline tab) -->
        <div v-if="activeTab === 'pipeline'" class="relative">
          <select
            v-model="selectedSymbol"
            @change="onSymbolChange"
            class="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 pr-8 text-white font-semibold focus:outline-none focus:border-blue-500 cursor-pointer appearance-none"
            :disabled="isLoading"
          >
            <option v-for="symbol in symbols" :key="symbol" :value="symbol">
              {{ symbol }}
            </option>
          </select>
          <!-- Dropdown Arrow -->
          <div
            class="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400"
          >
            ▼
          </div>
        </div>

        <!-- Connection Status -->
        <div class="flex items-center gap-2">
          <span
            class="w-3 h-3 rounded-full"
            :class="isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'"
          ></span>
          <span
            class="text-sm"
            :class="isConnected ? 'text-green-400' : 'text-red-400'"
          >
            {{ isConnected ? "Connected" : "Disconnected" }}
          </span>
        </div>

        <!-- Refresh Button -->
        <button
          @click="refreshAll"
          :disabled="isLoading"
          class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-semibold transition-colors flex items-center gap-2"
        >
          <span :class="{ 'animate-spin': isLoading }">🔄</span>
          Refresh
        </button>
      </div>
    </div>

    <!-- ==================== PIPELINE TAB ==================== -->
    <div v-if="activeTab === 'pipeline'">
      <!-- Main Grid: 3 Columns -->
      <div class="grid grid-cols-12 gap-4">
        <!-- LEFT: Bot Control & Signal (3 cols) -->
        <div class="col-span-3 space-y-4">
          <!-- Bot Control Card -->
          <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
              <span class="text-2xl">🤖</span> AI Bot Control
            </h2>

            <!-- Bot Status -->
            <div class="flex items-center justify-between mb-4">
              <span class="text-gray-400">Status:</span>
              <span
                class="px-3 py-1 rounded-full text-sm font-bold"
                :class="
                  botStatus.running
                    ? 'bg-green-500/20 text-green-400 border border-green-500/50'
                    : 'bg-gray-600/20 text-gray-400 border border-gray-500/50'
                "
              >
                {{ botStatus.running ? "🟢 RUNNING" : "⭕ STOPPED" }}
              </span>
            </div>

            <!-- Start/Stop Button -->
            <button
              @click="toggleBot"
              :disabled="isBotLoading"
              class="w-full py-3 rounded-lg font-bold text-lg transition-all"
              :class="
                botStatus.running
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              "
            >
              {{
                isBotLoading
                  ? "⏳ Processing..."
                  : botStatus.running
                    ? "🛑 Stop Bot"
                    : "🚀 Start Bot"
              }}
            </button>

            <!-- Bot Config -->
            <div class="mt-4 space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-400">Symbols:</span>
                <span class="text-white">{{
                  botStatus.config?.symbols?.join(", ") || "N/A"
                }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Timeframe:</span>
                <span class="text-white">{{
                  botStatus.config?.timeframe || "H1"
                }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Mode:</span>
                <span class="text-yellow-400">{{
                  botStatus.config?.quality || "HIGH"
                }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Interval:</span>
                <span class="text-white"
                  >{{ botStatus.config?.interval || 60 }}s</span
                >
              </div>
            </div>

            <!-- Quality Selector -->
            <div class="mt-4 pt-4 border-t border-gray-600">
              <div class="text-gray-400 text-xs mb-2">
                Signal Quality Filter
              </div>
              <div class="grid grid-cols-4 gap-1">
                <button
                  v-for="q in qualityLevels"
                  :key="q.value"
                  @click="setQuality(q.value)"
                  :disabled="!botStatus.running"
                  class="px-2 py-1 rounded text-xs font-semibold transition-all"
                  :class="[
                    botStatus.config?.quality === q.value
                      ? q.activeClass
                      : 'bg-gray-600 text-gray-300 hover:bg-gray-500',
                    !botStatus.running && 'opacity-50 cursor-not-allowed',
                  ]"
                >
                  {{ q.label }}
                </button>
              </div>
            </div>
          </div>

          <!-- Current Signal Card -->
          <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
              <span class="text-2xl">📡</span> Current Signal
            </h2>

            <div
              class="text-center p-4 rounded-lg mb-4"
              :class="getSignalBgClass(currentSignal.signal)"
            >
              <div class="text-4xl font-bold mb-1">
                {{ currentSignal.signal || "WAIT" }}
              </div>
              <div class="text-sm text-gray-300">
                {{ currentSignal.symbol || selectedSymbol }}
              </div>
            </div>

            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-400">Quality:</span>
                <span :class="getQualityClass(currentSignal.quality)">
                  {{ currentSignal.quality || "N/A" }}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Confidence:</span>
                <span class="text-white"
                  >{{ currentSignal.confidence?.toFixed(1) || 0 }}%</span
                >
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Entry:</span>
                <span class="text-white font-mono">{{
                  formatPrice(currentSignal.entry)
                }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Stop Loss:</span>
                <span class="text-red-400 font-mono">{{
                  formatPrice(currentSignal.stop_loss)
                }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Take Profit:</span>
                <span class="text-green-400 font-mono">{{
                  formatPrice(currentSignal.take_profit)
                }}</span>
              </div>
            </div>
          </div>

          <!-- Account Info -->
          <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
              <span class="text-2xl">💰</span> Account
            </h2>

            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-400">Balance:</span>
                <span class="text-white font-bold"
                  >${{ formatNumber(riskData.balance) }}</span
                >
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Equity:</span>
                <span class="text-white"
                  >${{ formatNumber(riskData.equity) }}</span
                >
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Daily P&L:</span>
                <span
                  :class="
                    riskData.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                  "
                >
                  {{ riskData.daily_pnl >= 0 ? "+" : ""
                  }}{{ riskData.daily_pnl?.toFixed(2) || 0 }}%
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Open Positions:</span>
                <span class="text-white"
                  >{{ riskData.open_positions || 0 }} /
                  {{ riskData.max_positions || 3 }}</span
                >
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Losing Streak:</span>
                <span
                  :class="
                    riskData.losing_streak > 2 ? 'text-red-400' : 'text-white'
                  "
                >
                  {{ riskData.losing_streak || 0 }} /
                  {{ riskData.max_losing_streak || 5 }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- CENTER: 16-Layer Pipeline (6 cols) -->
        <div class="col-span-6">
          <div
            class="bg-gray-800 rounded-xl p-4 border border-gray-700 relative"
          >
            <!-- Loading Overlay -->
            <div
              v-if="isLoading"
              class="absolute inset-0 bg-gray-800/80 rounded-xl flex items-center justify-center z-10"
            >
              <div class="text-center">
                <div class="text-4xl animate-spin mb-2">⚙️</div>
                <div class="text-gray-300">Loading {{ selectedSymbol }}...</div>
              </div>
            </div>

            <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
              <span class="text-2xl">🔬</span> 16-Layer Intelligence Pipeline
              <span
                class="text-sm font-normal px-2 py-0.5 rounded bg-blue-600/30 text-blue-400"
              >
                {{ selectedSymbol }}
              </span>
              <span class="text-xs text-gray-400 ml-auto">
                Last Update: {{ lastUpdateTime }}
              </span>
            </h2>

            <!-- Pipeline Flow -->
            <div class="space-y-2">
              <!-- Data Layer -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">📊</div>
                <div class="flex-1 bg-gray-700/50 rounded p-2">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">1. Data Lake</span>
                    <span :class="getLayerStatusClass(layers.data_lake)">
                      {{ layers.data_lake?.status || "READY" }}
                    </span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    {{ layers.data_lake?.candles || 0 }} candles loaded
                  </div>
                </div>
              </div>

              <!-- Pattern Layer -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🔍</div>
                <div class="flex-1 bg-gray-700/50 rounded p-2">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">2. Pattern Matcher (FAISS)</span>
                    <span :class="getLayerStatusClass(layers.pattern_matcher)">
                      {{ layers.pattern_matcher?.matches || 0 }} matches
                    </span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Similarity:
                    {{
                      (layers.pattern_matcher?.similarity * 100)?.toFixed(1) ||
                      0
                    }}%
                  </div>
                </div>
              </div>

              <!-- Voting System -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🗳️</div>
                <div class="flex-1 bg-gray-700/50 rounded p-2">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">3. Voting System</span>
                    <span :class="getSignalClass(layers.voting?.signal)">
                      {{ layers.voting?.signal || "WAIT" }}
                    </span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Bullish: {{ layers.voting?.bullish || 0 }} | Bearish:
                    {{ layers.voting?.bearish || 0 }}
                  </div>
                </div>
              </div>

              <!-- Enhanced Analyzer -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🎯</div>
                <div class="flex-1 bg-gray-700/50 rounded p-2">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">4. Enhanced Analyzer</span>
                    <span :class="getQualityClass(layers.enhanced?.quality)">
                      {{ layers.enhanced?.quality || "N/A" }}
                    </span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Score: {{ layers.enhanced?.score?.toFixed(1) || 0 }}%
                  </div>
                </div>
              </div>

              <!-- Divider -->
              <div class="border-t border-gray-600 my-2"></div>
              <div class="text-xs text-center text-gray-500 -my-1">
                ⬇️ Deep Intelligence Modules ⬇️
              </div>

              <!-- Advanced Intelligence -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🧠</div>
                <div
                  class="flex-1 bg-blue-900/30 rounded p-2 border border-blue-500/30"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium">5. Advanced Intelligence</span>
                    <span class="text-blue-400">{{
                      layers.advanced?.regime || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    MTF: {{ layers.advanced?.mtf_alignment || "N/A" }} |
                    Multiplier:
                    {{ layers.advanced?.multiplier?.toFixed(2) || 1 }}x
                  </div>
                </div>
              </div>

              <!-- Smart Brain -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">💡</div>
                <div
                  class="flex-1 bg-blue-900/30 rounded p-2 border border-blue-500/30"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium">6. Smart Brain</span>
                    <span class="text-blue-400"
                      >{{ layers.smart?.pattern_count || 0 }} patterns</span
                    >
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Adaptive Risk:
                    {{ layers.smart?.multiplier?.toFixed(2) || 1 }}x
                  </div>
                </div>
              </div>

              <!-- Neural Brain -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🧬</div>
                <div
                  class="flex-1 bg-purple-900/30 rounded p-2 border border-purple-500/30"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium"
                      >7. Neural Brain (Pattern DNA)</span
                    >
                    <span class="text-purple-400">{{
                      layers.neural?.market_state || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    DNA Score: {{ layers.neural?.dna_score?.toFixed(1) || 0 }}%
                    | Multiplier:
                    {{ layers.neural?.multiplier?.toFixed(2) || 1 }}x
                  </div>
                </div>
              </div>

              <!-- Deep Intelligence -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🔮</div>
                <div
                  class="flex-1 bg-purple-900/30 rounded p-2 border border-purple-500/30"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium"
                      >8. Deep Intelligence (Cross-Asset)</span
                    >
                    <span class="text-purple-400"
                      >{{ (layers.deep?.correlation * 100)?.toFixed(0) || 0 }}%
                      corr</span
                    >
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Session: {{ layers.deep?.session || "N/A" }} | Multiplier:
                    {{ layers.deep?.multiplier?.toFixed(2) || 1 }}x
                  </div>
                </div>
              </div>

              <!-- Quantum Strategy -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">⚛️</div>
                <div
                  class="flex-1 bg-cyan-900/30 rounded p-2 border border-cyan-500/30"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium"
                      >9. Quantum Strategy (Microstructure)</span
                    >
                    <span class="text-cyan-400">{{
                      layers.quantum?.volatility_regime || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Fractal: {{ layers.quantum?.fractal || "N/A" }} |
                    Multiplier:
                    {{ layers.quantum?.multiplier?.toFixed(2) || 1 }}x
                  </div>
                </div>
              </div>

              <!-- Divider -->
              <div class="border-t border-gray-600 my-2"></div>
              <div class="text-xs text-center text-gray-500 -my-1">
                ⬇️ Professional Grade Modules ⬇️
              </div>

              <!-- Alpha Engine -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🔶</div>
                <div
                  class="flex-1 bg-orange-900/30 rounded p-2 border border-orange-500/30"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium">10. Alpha Engine</span>
                    <span class="text-orange-400 font-bold">{{
                      alphaData.grade || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Score: {{ alphaData.alpha_score?.toFixed(1) || 0 }}% | Flow:
                    {{ alphaData.order_flow_bias || "N/A" }}
                  </div>
                  <div class="text-xs text-gray-400">
                    R:R {{ alphaData.risk_reward?.toFixed(1) || 0 }} |
                    Multiplier:
                    {{ alphaData.position_multiplier?.toFixed(2) || 1 }}x
                  </div>
                </div>
              </div>

              <!-- Omega Brain -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">⚡</div>
                <div
                  class="flex-1 bg-indigo-900/30 rounded p-2 border border-indigo-500/30"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium"
                      >11. Omega Brain (Institutional)</span
                    >
                    <span class="text-indigo-400 font-bold">{{
                      omegaData.grade || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Score: {{ omegaData.omega_score?.toFixed(1) || 0 }}% | Flow:
                    {{ omegaData.institutional_flow || "N/A" }}
                  </div>
                  <div class="text-xs text-gray-400">
                    Smart Money: {{ omegaData.smart_money || "N/A" }} |
                    Manipulation:
                    {{ omegaData.manipulation_detected || "NONE" }}
                  </div>
                </div>
              </div>

              <!-- Divider -->
              <div class="border-t border-gray-600 my-2"></div>
              <div class="text-xs text-center text-gray-500 -my-1">
                ⬇️ Meta-Intelligence Synthesis ⬇️
              </div>

              <!-- Titan Core -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🏛️</div>
                <div
                  class="flex-1 bg-gradient-to-r from-purple-900/50 to-yellow-900/50 rounded p-2 border border-yellow-500/50"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium"
                      >12. TITAN CORE (Meta-Intelligence)</span
                    >
                    <span class="text-yellow-400 font-bold text-lg">{{
                      titanData.grade || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Score: {{ titanData.titan_score?.toFixed(1) || 0 }}% |
                    Consensus: {{ titanData.consensus || "N/A" }}
                  </div>
                  <div class="text-xs text-gray-400">
                    Prediction:
                    {{ titanData.prediction?.direction || "WAIT" }} ({{
                      titanData.prediction?.predicted_move?.toFixed(2) || 0
                    }}%)
                  </div>
                  <div class="text-xs text-gray-400">
                    Modules: {{ titanData.agreeing_modules || 0 }}/{{
                      titanData.total_modules || 0
                    }}
                    agree | Multiplier:
                    {{ titanData.position_multiplier?.toFixed(2) || 1 }}x
                  </div>
                </div>
              </div>

              <!-- Divider -->
              <div class="border-t border-gray-600 my-2"></div>
              <div class="text-xs text-center text-gray-500 -my-1">
                ⬇️ Risk & Protection Layer ⬇️
              </div>

              <!-- Continuous Learning -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">📚</div>
                <div class="flex-1 bg-gray-700/50 rounded p-2">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">13. Continuous Learning</span>
                    <span class="text-gray-400">{{
                      layers.learning?.market_cycle || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Cycles: {{ layers.learning?.cycles || 0 }} | Adaptations:
                    {{ layers.learning?.adaptations || 0 }}
                  </div>
                </div>
              </div>

              <!-- Pro Features -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">⭐</div>
                <div class="flex-1 bg-gray-700/50 rounded p-2">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">14. Pro Features (Sessions)</span>
                    <span class="text-gray-400">{{
                      layers.pro?.session || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    News Impact: {{ layers.pro?.news_impact || "NONE" }} |
                    Multiplier: {{ layers.pro?.multiplier?.toFixed(2) || 1 }}x
                  </div>
                </div>
              </div>

              <!-- Risk Guardian -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">🛡️</div>
                <div
                  class="flex-1 rounded p-2 border"
                  :class="getRiskBgClass(riskData.risk_level)"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium">15. Risk Guardian</span>
                    <span
                      class="font-bold"
                      :class="getRiskTextClass(riskData.risk_level)"
                    >
                      {{ riskData.risk_level || "SAFE" }}
                    </span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Can Trade: {{ riskData.can_trade ? "✅ YES" : "❌ NO" }} |
                    Daily: {{ riskData.daily_pnl?.toFixed(2) || 0 }}% / -{{
                      riskData.max_daily_loss || 5
                    }}%
                  </div>
                </div>
              </div>

              <!-- Sentiment Analyzer (Contrarian) -->
              <div class="flex items-center gap-2">
                <div class="w-8 text-center text-lg">😈</div>
                <div
                  class="flex-1 bg-pink-900/30 rounded p-2 border border-pink-500/30"
                >
                  <div class="flex justify-between items-center">
                    <span class="font-medium"
                      >16. Contrarian (Anti-Retail)</span
                    >
                    <span class="text-pink-400">{{
                      layers.sentiment?.level || "N/A"
                    }}</span>
                  </div>
                  <div class="text-xs text-gray-400 mt-1">
                    Retail Sentiment:
                    {{ layers.sentiment?.retail_sentiment || 0 }}% | Override:
                    {{ layers.sentiment?.override || "NO" }}
                  </div>
                </div>
              </div>

              <!-- Final Decision -->
              <div
                class="mt-4 p-4 rounded-lg bg-gradient-to-r from-gray-700 to-gray-800 border-2"
                :class="getFinalDecisionBorderClass(finalDecision.action)"
              >
                <div class="flex items-center justify-between">
                  <div>
                    <div class="text-sm text-gray-400">FINAL DECISION</div>
                    <div
                      class="text-2xl font-bold"
                      :class="getFinalDecisionClass(finalDecision.action)"
                    >
                      {{ finalDecision.action || "WAITING" }}
                    </div>
                  </div>
                  <div class="text-right">
                    <div class="text-sm text-gray-400">Position Size</div>
                    <div class="text-xl font-bold text-white">
                      {{
                        finalDecision.position_multiplier?.toFixed(2) || 1.0
                      }}x
                    </div>
                  </div>
                </div>
                <div class="mt-2 text-xs text-gray-400">
                  {{ finalDecision.verdict || "Waiting for analysis..." }}
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- RIGHT: Positions & History (3 cols) -->
        <div class="col-span-3 space-y-4">
          <!-- Open Positions -->
          <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
              <span class="text-2xl">📊</span> Open Positions
              <span class="text-xs bg-gray-600 px-2 py-1 rounded ml-auto">
                {{ positions.length }}
              </span>
            </h2>

            <div
              v-if="positions.length === 0"
              class="text-center text-gray-400 py-8"
            >
              No open positions
            </div>

            <div v-else class="space-y-2 max-h-60 overflow-y-auto">
              <div
                v-for="pos in positions"
                :key="pos.id"
                class="p-3 rounded bg-gray-700/50 border"
                :class="
                  pos.pnl >= 0 ? 'border-green-500/30' : 'border-red-500/30'
                "
              >
                <div class="flex justify-between items-center">
                  <div>
                    <span class="font-bold">{{ pos.symbol }}</span>
                    <span
                      class="ml-2 text-xs px-2 py-0.5 rounded"
                      :class="
                        pos.side === 'BUY'
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                      "
                    >
                      {{ pos.side }}
                    </span>
                  </div>
                  <span
                    class="font-bold"
                    :class="pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'"
                  >
                    {{ pos.pnl >= 0 ? "+" : "" }}${{ pos.pnl?.toFixed(2) || 0 }}
                  </span>
                </div>
                <div class="text-xs text-gray-400 mt-1 flex justify-between">
                  <span>Qty: {{ pos.quantity }}</span>
                  <span>Entry: {{ pos.entry_price?.toFixed(5) }}</span>
                </div>
                <div class="text-xs text-gray-400 flex justify-between">
                  <span class="text-red-400"
                    >SL: {{ pos.stop_loss?.toFixed(5) || "N/A" }}</span
                  >
                  <span class="text-green-400"
                    >TP: {{ pos.take_profit?.toFixed(5) || "N/A" }}</span
                  >
                </div>
              </div>
            </div>
          </div>

          <!-- Trade History -->
          <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
              <span class="text-2xl">📜</span> Recent Trades
            </h2>

            <div
              v-if="tradeHistory.length === 0"
              class="text-center text-gray-400 py-8"
            >
              No trades yet
            </div>

            <div v-else class="space-y-2 max-h-80 overflow-y-auto">
              <div
                v-for="trade in tradeHistory"
                :key="trade.id"
                class="p-2 rounded bg-gray-700/50 text-sm"
              >
                <div class="flex justify-between items-center">
                  <span class="font-medium">{{ trade.symbol }}</span>
                  <span
                    class="font-bold"
                    :class="trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'"
                  >
                    {{ trade.pnl >= 0 ? "+" : "" }}${{
                      trade.pnl?.toFixed(2) || 0
                    }}
                  </span>
                </div>
                <div class="text-xs text-gray-400 flex justify-between mt-1">
                  <span>{{ trade.side }} {{ trade.quantity }}</span>
                  <span>{{ formatTime(trade.closed_at) }}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Signal History -->
          <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
              <span class="text-2xl">📡</span> Signal History
            </h2>

            <div
              v-if="signalHistory.length === 0"
              class="text-center text-gray-400 py-4"
            >
              No signals yet
            </div>

            <div v-else class="space-y-2 max-h-40 overflow-y-auto">
              <div
                v-for="sig in signalHistory.slice(0, 10)"
                :key="sig.timestamp"
                class="flex justify-between items-center text-sm p-2 rounded bg-gray-700/30"
              >
                <span class="text-gray-400">{{ sig.symbol }}</span>
                <span :class="getSignalClass(sig.signal)">{{
                  sig.signal
                }}</span>
                <span class="text-xs text-gray-500">{{
                  formatTime(sig.timestamp)
                }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Footer Stats -->
      <div class="mt-4 grid grid-cols-6 gap-4">
        <div
          class="bg-gray-800 rounded-lg p-3 text-center border border-gray-700"
        >
          <div class="text-xs text-gray-400">Total Trades</div>
          <div class="text-xl font-bold text-white">
            {{ stats.total_trades || 0 }}
          </div>
        </div>
        <div
          class="bg-gray-800 rounded-lg p-3 text-center border border-gray-700"
        >
          <div class="text-xs text-gray-400">Win Rate</div>
          <div class="text-xl font-bold text-green-400">
            {{ stats.win_rate?.toFixed(1) || 0 }}%
          </div>
        </div>
        <div
          class="bg-gray-800 rounded-lg p-3 text-center border border-gray-700"
        >
          <div class="text-xs text-gray-400">Profit Factor</div>
          <div class="text-xl font-bold text-blue-400">
            {{ stats.profit_factor?.toFixed(2) || 0 }}
          </div>
        </div>
        <div
          class="bg-gray-800 rounded-lg p-3 text-center border border-gray-700"
        >
          <div class="text-xs text-gray-400">Total P&L</div>
          <div
            class="text-xl font-bold"
            :class="stats.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'"
          >
            ${{ formatNumber(stats.total_pnl || 0) }}
          </div>
        </div>
        <div
          class="bg-gray-800 rounded-lg p-3 text-center border border-gray-700"
        >
          <div class="text-xs text-gray-400">Avg Win</div>
          <div class="text-xl font-bold text-green-400">
            ${{ formatNumber(stats.avg_win || 0) }}
          </div>
        </div>
        <div
          class="bg-gray-800 rounded-lg p-3 text-center border border-gray-700"
        >
          <div class="text-xs text-gray-400">Avg Loss</div>
          <div class="text-xl font-bold text-red-400">
            ${{ formatNumber(stats.avg_loss || 0) }}
          </div>
        </div>
      </div>
    </div>
    <!-- END PIPELINE TAB -->

    <!-- ==================== SYSTEM HEALTH TAB ==================== -->
    <div v-if="activeTab === 'health'" class="space-y-6">
      <!-- Health Overview Cards -->
      <div class="grid grid-cols-4 gap-4">
        <!-- MT5 Connection -->
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div class="flex items-center justify-between mb-4">
            <span class="text-2xl">🔌</span>
            <span
              class="w-4 h-4 rounded-full"
              :class="
                systemHealth.mt5_connected
                  ? 'bg-green-500 animate-pulse'
                  : 'bg-red-500'
              "
            ></span>
          </div>
          <h3 class="text-lg font-bold text-white">MT5 Connection</h3>
          <p
            :class="
              systemHealth.mt5_connected ? 'text-green-400' : 'text-red-400'
            "
          >
            {{
              systemHealth.mt5_connected ? "✅ Connected" : "❌ Disconnected"
            }}
          </p>
          <p class="text-xs text-gray-500 mt-2">Broker: Exness-MT5Real39</p>
        </div>

        <!-- API Server -->
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div class="flex items-center justify-between mb-4">
            <span class="text-2xl">🖥️</span>
            <span
              class="w-4 h-4 rounded-full"
              :class="
                systemHealth.api_status === 'online'
                  ? 'bg-green-500 animate-pulse'
                  : 'bg-red-500'
              "
            ></span>
          </div>
          <h3 class="text-lg font-bold text-white">API Server</h3>
          <p
            :class="
              systemHealth.api_status === 'online'
                ? 'text-green-400'
                : 'text-red-400'
            "
          >
            {{
              systemHealth.api_status === "online" ? "✅ Online" : "❌ Offline"
            }}
          </p>
          <p class="text-xs text-gray-500 mt-2">Port: 8000</p>
        </div>

        <!-- Bot Status -->
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div class="flex items-center justify-between mb-4">
            <span class="text-2xl">🤖</span>
            <span
              class="w-4 h-4 rounded-full"
              :class="
                systemHealth.bot_running
                  ? 'bg-green-500 animate-pulse'
                  : 'bg-yellow-500'
              "
            ></span>
          </div>
          <h3 class="text-lg font-bold text-white">AI Bot</h3>
          <p
            :class="
              systemHealth.bot_running ? 'text-green-400' : 'text-yellow-400'
            "
          >
            {{ systemHealth.bot_running ? "✅ Running" : "⏸️ Stopped" }}
          </p>
          <p class="text-xs text-gray-500 mt-2">
            Quality: {{ botStatus.config?.quality || "N/A" }}
          </p>
        </div>

        <!-- Intelligence Modules -->
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div class="flex items-center justify-between mb-4">
            <span class="text-2xl">🧠</span>
            <span class="text-lg font-bold text-blue-400">
              {{ systemHealth.intelligence_modules }}/{{
                systemHealth.total_modules
              }}
            </span>
          </div>
          <h3 class="text-lg font-bold text-white">Intelligence</h3>
          <p class="text-blue-400">
            {{
              Math.round(
                (systemHealth.intelligence_modules /
                  systemHealth.total_modules) *
                  100,
              )
            }}% Active
          </p>
          <p class="text-xs text-gray-500 mt-2">16-Layer Analysis</p>
        </div>
      </div>

      <!-- Detailed Health Checks -->
      <div class="grid grid-cols-2 gap-6">
        <!-- System Components -->
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>⚙️</span> System Components
          </h3>
          <div class="space-y-3">
            <div
              class="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
            >
              <span class="text-gray-300">Data Lake</span>
              <span
                :class="
                  systemHealth.data_lake_ready
                    ? 'text-green-400'
                    : 'text-red-400'
                "
              >
                {{ systemHealth.data_lake_ready ? "✅ Ready" : "❌ Not Ready" }}
              </span>
            </div>
            <div
              class="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
            >
              <span class="text-gray-300">FAISS Index</span>
              <span
                :class="
                  systemHealth.faiss_loaded
                    ? 'text-green-400'
                    : 'text-yellow-400'
                "
              >
                {{ systemHealth.faiss_loaded ? "✅ Loaded" : "⏳ Loading..." }}
              </span>
            </div>
            <div
              class="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
            >
              <span class="text-gray-300">Pattern Matcher</span>
              <span
                :class="
                  layers.pattern_matcher?.status === 'ACTIVE'
                    ? 'text-green-400'
                    : 'text-yellow-400'
                "
              >
                {{ layers.pattern_matcher?.status || "N/A" }}
              </span>
            </div>
            <div
              class="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
            >
              <span class="text-gray-300">Voting System</span>
              <span class="text-green-400">✅ Active</span>
            </div>
            <div
              class="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
            >
              <span class="text-gray-300">Enhanced Analyzer</span>
              <span class="text-green-400">✅ Active</span>
            </div>
          </div>
        </div>

        <!-- Intelligence Modules Status -->
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>🧠</span> Intelligence Modules
          </h3>
          <div class="space-y-2 max-h-80 overflow-y-auto">
            <div
              v-for="(module, idx) in intelligenceModules"
              :key="idx"
              class="flex items-center justify-between p-2 bg-gray-700/30 rounded"
            >
              <div class="flex items-center gap-2">
                <span>{{ module.icon }}</span>
                <span class="text-sm text-gray-300">{{ module.name }}</span>
              </div>
              <span
                class="text-xs px-2 py-1 rounded"
                :class="
                  module.active
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-gray-600 text-gray-400'
                "
              >
                {{ module.active ? "ACTIVE" : "INACTIVE" }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Performance Metrics -->
      <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <span>📊</span> Performance Metrics
        </h3>
        <div class="grid grid-cols-4 gap-6">
          <div class="text-center">
            <div class="text-3xl font-bold text-blue-400">
              {{ systemHealth.last_analysis_time || "N/A" }}
            </div>
            <div class="text-sm text-gray-400 mt-1">Last Analysis</div>
          </div>
          <div class="text-center">
            <div class="text-3xl font-bold text-green-400">
              {{ formatUptime(systemHealth.uptime) }}
            </div>
            <div class="text-sm text-gray-400 mt-1">Uptime</div>
          </div>
          <div class="text-center">
            <div class="text-3xl font-bold text-purple-400">
              {{ layers.data_lake?.candles || 0 }}
            </div>
            <div class="text-sm text-gray-400 mt-1">Candles Loaded</div>
          </div>
          <div class="text-center">
            <div class="text-3xl font-bold text-yellow-400">
              {{ layers.pattern_matcher?.matches || 0 }}
            </div>
            <div class="text-sm text-gray-400 mt-1">Pattern Matches</div>
          </div>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <span>⚡</span> Quick Actions
        </h3>
        <div class="flex gap-4">
          <button
            @click="checkSystemHealth"
            class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
          >
            🔄 Refresh Health Check
          </button>
          <button
            @click="toggleBot"
            :disabled="isBotLoading"
            class="px-6 py-3 rounded-lg font-semibold transition-colors"
            :class="
              botStatus.running
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-green-600 hover:bg-green-700 text-white'
            "
          >
            {{ botStatus.running ? "🛑 Stop Bot" : "🚀 Start Bot" }}
          </button>
          <button
            @click="rebuildIndex"
            :disabled="isLoading"
            class="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors"
          >
            🔧 Rebuild FAISS Index
          </button>
        </div>
      </div>
    </div>
    <!-- END SYSTEM HEALTH TAB -->
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import api from "@/services/api";

// State
const symbols = ref(["EURUSDm", "GBPUSDm", "XAUUSDm"]);
const selectedSymbol = ref("EURUSDm");
const isLoading = ref(false);
const isBotLoading = ref(false);
const isConnected = ref(false);
const lastUpdateTime = ref("--:--:--");
const activeTab = ref("pipeline"); // pipeline, health

// Quality Levels
const qualityLevels = [
  {
    value: "LOW",
    label: "LOW",
    confidence: 50,
    activeClass: "bg-gray-500 text-white border-2 border-gray-400",
  },
  {
    value: "MEDIUM",
    label: "MEDIUM",
    confidence: 65,
    activeClass: "bg-yellow-600 text-white border-2 border-yellow-400",
  },
  {
    value: "HIGH",
    label: "HIGH",
    confidence: 75,
    activeClass: "bg-blue-600 text-white border-2 border-blue-400",
  },
  {
    value: "PREMIUM",
    label: "PREMIUM",
    confidence: 85,
    activeClass: "bg-purple-600 text-white border-2 border-purple-400",
  },
];

// System Health
const systemHealth = ref({
  mt5_connected: false,
  api_status: "unknown",
  bot_running: false,
  data_lake_ready: false,
  faiss_loaded: false,
  intelligence_modules: 0,
  total_modules: 16,
  last_analysis_time: null,
  memory_usage: 0,
  uptime: 0,
});

// Bot Status
const botStatus = ref({
  running: false,
  config: {},
});

// Current Signal
const currentSignal = ref({
  signal: "WAIT",
  symbol: "",
  quality: null,
  confidence: 0,
  entry: null,
  stop_loss: null,
  take_profit: null,
});

// Intelligence Data
const titanData = ref({});
const omegaData = ref({});
const alphaData = ref({});
const riskData = ref({
  risk_level: "SAFE",
  balance: 0,
  equity: 0,
  daily_pnl: 0,
  open_positions: 0,
  max_positions: 3,
  can_trade: true,
  losing_streak: 0,
  max_losing_streak: 5,
  max_daily_loss: 5,
});

// Layer statuses
const layers = ref({
  data_lake: { status: "READY", candles: 0 },
  pattern_matcher: { matches: 0, similarity: 0 },
  voting: { signal: "WAIT", bullish: 0, bearish: 0 },
  enhanced: { quality: null, score: 0 },
  advanced: { regime: null, mtf_alignment: null, multiplier: 1 },
  smart: { pattern_count: 0, multiplier: 1 },
  neural: { market_state: null, dna_score: 0, multiplier: 1 },
  deep: { correlation: 0, session: null, multiplier: 1 },
  quantum: { volatility_regime: null, fractal: null, multiplier: 1 },
  learning: { market_cycle: null, cycles: 0, adaptations: 0 },
  pro: { session: null, news_impact: null, multiplier: 1 },
  sentiment: { level: null, retail_sentiment: 0, override: "NO" },
});

// Intelligence Modules for Health Display
const intelligenceModules = computed(() => [
  { name: "Data Lake", icon: "📊", active: systemHealth.value.data_lake_ready },
  {
    name: "Pattern Matcher (FAISS)",
    icon: "🔍",
    active: systemHealth.value.faiss_loaded,
  },
  { name: "Voting System", icon: "🗳️", active: true },
  { name: "Enhanced Analyzer", icon: "🎯", active: true },
  {
    name: "Advanced Intelligence",
    icon: "🧠",
    active: layers.value.advanced?.regime !== null,
  },
  {
    name: "Smart Brain",
    icon: "💡",
    active: layers.value.smart?.pattern_count > 0,
  },
  {
    name: "Neural Brain",
    icon: "🧬",
    active: layers.value.neural?.market_state !== null,
  },
  {
    name: "Deep Intelligence",
    icon: "🌊",
    active: layers.value.deep?.session !== null,
  },
  {
    name: "Quantum Strategy",
    icon: "⚛️",
    active: layers.value.quantum?.volatility_regime !== null,
  },
  {
    name: "Alpha Engine",
    icon: "🚀",
    active: alphaData.value?.grade !== "N/A",
  },
  { name: "Omega Brain", icon: "🌟", active: omegaData.value?.grade !== "N/A" },
  { name: "Titan Core", icon: "⚔️", active: titanData.value?.grade !== "N/A" },
  {
    name: "Continuous Learning",
    icon: "📚",
    active: layers.value.learning?.cycles > 0,
  },
  {
    name: "Pro Features",
    icon: "🏆",
    active: layers.value.pro?.session !== null,
  },
  {
    name: "Risk Guardian",
    icon: "🛡️",
    active: riskData.value.can_trade !== undefined,
  },
  {
    name: "Sentiment Analyzer",
    icon: "📰",
    active: layers.value.sentiment?.level !== null,
  },
]);

// Final Decision
const finalDecision = ref({
  action: "WAITING",
  position_multiplier: 1,
  verdict: "Waiting for analysis...",
});

// Positions & History
const positions = ref([]);
const tradeHistory = ref([]);
const signalHistory = ref([]);
const stats = ref({
  total_trades: 0,
  win_rate: 0,
  profit_factor: 0,
  total_pnl: 0,
  avg_win: 0,
  avg_loss: 0,
});

// Refresh interval
let refreshInterval = null;

// Methods
const refreshAll = async () => {
  isLoading.value = true;
  try {
    // Parallel fetch all data
    const [botStatusRes, pipelineRes, riskRes, positionsRes, historyRes] =
      await Promise.all([
        api.getBotStatus(),
        api.getPipelineData(selectedSymbol.value),
        api.getRiskData(),
        api.getPositions(),
        api.getTradeHistory(20),
      ]);

    // Check connection
    isConnected.value = !botStatusRes._isMock;

    // Update Bot Status
    if (botStatusRes && !botStatusRes._isMock) {
      botStatus.value = {
        running: botStatusRes.running || false,
        config: botStatusRes.config || {},
      };
    }

    // Update from Pipeline API (comprehensive data)
    if (
      pipelineRes &&
      !pipelineRes._isMock &&
      pipelineRes.status === "active"
    ) {
      // Update all layers
      if (pipelineRes.layers) {
        const pl = pipelineRes.layers;

        // Data Lake
        if (pl.data_lake) {
          layers.value.data_lake = pl.data_lake;
        }

        // Pattern Matcher
        if (pl.pattern_matcher) {
          layers.value.pattern_matcher = pl.pattern_matcher;
        }

        // Voting
        if (pl.voting) {
          layers.value.voting = pl.voting;
        }

        // Enhanced
        if (pl.enhanced) {
          layers.value.enhanced = pl.enhanced;
        }

        // Advanced Intelligence
        if (pl.advanced) {
          layers.value.advanced = pl.advanced;
        }

        // Smart Brain
        if (pl.smart) {
          layers.value.smart = pl.smart;
        }

        // Neural Brain
        if (pl.neural) {
          layers.value.neural = pl.neural;
        }

        // Deep Intelligence
        if (pl.deep) {
          layers.value.deep = pl.deep;
        }

        // Quantum Strategy
        if (pl.quantum) {
          layers.value.quantum = pl.quantum;
        }

        // Alpha Engine
        if (pl.alpha) {
          alphaData.value = pl.alpha;
        }

        // Omega Brain
        if (pl.omega) {
          omegaData.value = pl.omega;
        }

        // Titan Core
        if (pl.titan) {
          titanData.value = pl.titan;
        }

        // Continuous Learning
        if (pl.learning) {
          layers.value.learning = pl.learning;
        }

        // Pro Features
        if (pl.pro) {
          layers.value.pro = pl.pro;
        }

        // Risk Guardian
        if (pl.risk) {
          riskData.value = { ...riskData.value, ...pl.risk };
        }

        // Sentiment
        if (pl.sentiment) {
          layers.value.sentiment = pl.sentiment;
        }
      }

      // Update Current Signal
      if (pipelineRes.current_signal) {
        currentSignal.value = pipelineRes.current_signal;
      }

      // Update Final Decision
      if (pipelineRes.final_decision) {
        finalDecision.value = {
          action: pipelineRes.final_decision.signal || "WAITING",
          position_multiplier:
            pipelineRes.final_decision.position_multiplier || 1,
          verdict: pipelineRes.final_decision.verdict || "",
        };
      }
    }

    // Fallback: Update Risk from separate API if needed
    if (riskRes && !riskRes._isMock) {
      riskData.value = { ...riskData.value, ...riskRes };
    }

    // Update Positions
    if (positionsRes && positionsRes.positions) {
      positions.value = positionsRes.positions;
    }

    // Update Trade History
    if (historyRes && historyRes.trades) {
      tradeHistory.value = historyRes.trades;

      // Calculate stats
      const wins = historyRes.trades.filter((t) => t.pnl > 0);
      const losses = historyRes.trades.filter((t) => t.pnl < 0);
      stats.value = {
        total_trades: historyRes.trades.length,
        win_rate:
          historyRes.trades.length > 0
            ? (wins.length / historyRes.trades.length) * 100
            : 0,
        profit_factor:
          losses.reduce((s, t) => s + Math.abs(t.pnl), 0) > 0
            ? wins.reduce((s, t) => s + t.pnl, 0) /
              losses.reduce((s, t) => s + Math.abs(t.pnl), 0)
            : 0,
        total_pnl: historyRes.trades.reduce((s, t) => s + (t.pnl || 0), 0),
        avg_win:
          wins.length > 0
            ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length
            : 0,
        avg_loss:
          losses.length > 0
            ? losses.reduce((s, t) => s + Math.abs(t.pnl), 0) / losses.length
            : 0,
      };
    }

    // Signal history from bot signals
    const signalsRes = await api.getBotSignals(20);
    if (signalsRes && signalsRes.signals) {
      signalHistory.value = signalsRes.signals;
    }

    lastUpdateTime.value = new Date().toLocaleTimeString();
  } catch (error) {
    console.error("Refresh failed:", error);
    isConnected.value = false;
  } finally {
    isLoading.value = false;
  }
};

const toggleBot = async () => {
  isBotLoading.value = true;
  try {
    if (botStatus.value.running) {
      await api.stopBot();
    } else {
      await api.startBot({
        symbols: symbols.value,
        timeframe: "H1",
        quality: "HIGH",
        interval: 60,
      });
    }
    // Refresh status
    await refreshAll();
  } catch (error) {
    console.error("Bot toggle failed:", error);
  } finally {
    isBotLoading.value = false;
  }
};

// Helper functions
const formatPrice = (price) => {
  if (!price) return "N/A";
  return price.toFixed(5);
};

const formatNumber = (num) => {
  if (!num) return "0";
  return num.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
};

const formatTime = (timestamp) => {
  if (!timestamp) return "--:--";
  const date = new Date(timestamp);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
};

const getSignalBgClass = (signal) => {
  if (!signal) return "bg-gray-700";
  if (signal === "STRONG_BUY" || signal === "BUY")
    return "bg-green-900/50 border border-green-500/50";
  if (signal === "STRONG_SELL" || signal === "SELL")
    return "bg-red-900/50 border border-red-500/50";
  return "bg-gray-700 border border-gray-500/50";
};

const getSignalClass = (signal) => {
  if (!signal) return "text-gray-400";
  if (signal === "STRONG_BUY" || signal === "BUY") return "text-green-400";
  if (signal === "STRONG_SELL" || signal === "SELL") return "text-red-400";
  return "text-yellow-400";
};

const getQualityClass = (quality) => {
  if (!quality) return "text-gray-400";
  if (quality === "PREMIUM") return "text-purple-400 font-bold";
  if (quality === "HIGH") return "text-green-400 font-bold";
  if (quality === "MEDIUM") return "text-yellow-400";
  return "text-gray-400";
};

const getLayerStatusClass = (layer) => {
  if (!layer) return "text-gray-400";
  if (layer.status === "READY" || layer.status === "ACTIVE")
    return "text-green-400";
  if (layer.status === "ERROR") return "text-red-400";
  return "text-yellow-400";
};

const getRiskBgClass = (level) => {
  if (level === "SAFE") return "bg-green-900/30 border-green-500/50";
  if (level === "CAUTION") return "bg-yellow-900/30 border-yellow-500/50";
  if (level === "WARNING") return "bg-orange-900/30 border-orange-500/50";
  if (level === "DANGER") return "bg-red-900/30 border-red-500/50";
  return "bg-gray-700/30 border-gray-500/50";
};

const getRiskTextClass = (level) => {
  if (level === "SAFE") return "text-green-400";
  if (level === "CAUTION") return "text-yellow-400";
  if (level === "WARNING") return "text-orange-400";
  if (level === "DANGER") return "text-red-400";
  return "text-gray-400";
};

const getFinalDecisionClass = (action) => {
  if (action === "STRONG_BUY" || action === "BUY") return "text-green-400";
  if (action === "STRONG_SELL" || action === "SELL") return "text-red-400";
  if (action === "BLOCKED") return "text-orange-400";
  return "text-yellow-400";
};

const getFinalDecisionBorderClass = (action) => {
  if (action === "STRONG_BUY" || action === "BUY") return "border-green-500";
  if (action === "STRONG_SELL" || action === "SELL") return "border-red-500";
  if (action === "BLOCKED") return "border-orange-500";
  return "border-yellow-500";
};

// Format uptime
const formatUptime = (seconds) => {
  if (!seconds) return "0s";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
};

// Set Quality
const setQuality = async (quality) => {
  try {
    await api.updateBotSettings({ min_quality: quality });
    botStatus.value.config.quality = quality;
    console.log(`✅ Quality updated to ${quality}`);
  } catch (error) {
    console.error("Failed to update quality:", error);
  }
};

// Check System Health
const checkSystemHealth = async () => {
  isLoading.value = true;
  try {
    // Fetch health data
    const [healthRes, botStatusRes] = await Promise.all([
      api.getSystemHealth().catch(() => ({})),
      api.getBotStatus(),
    ]);

    // Update system health
    systemHealth.value = {
      mt5_connected: healthRes.mt5_connected ?? botStatusRes.running,
      api_status:
        healthRes.api_status ?? (botStatusRes._isMock ? "offline" : "online"),
      bot_running: botStatusRes.running || false,
      data_lake_ready:
        healthRes.data_lake_ready ?? layers.value.data_lake?.status === "READY",
      faiss_loaded:
        healthRes.faiss_loaded ??
        layers.value.pattern_matcher?.status === "ACTIVE",
      intelligence_modules:
        healthRes.intelligence_modules ??
        intelligenceModules.value.filter((m) => m.active).length,
      total_modules: 16,
      last_analysis_time: healthRes.last_analysis_time ?? lastUpdateTime.value,
      memory_usage: healthRes.memory_usage ?? 0,
      uptime: healthRes.uptime ?? 0,
    };

    isConnected.value = systemHealth.value.api_status === "online";
  } catch (error) {
    console.error("Health check failed:", error);
  } finally {
    isLoading.value = false;
  }
};

// Rebuild FAISS Index
const rebuildIndex = async () => {
  isLoading.value = true;
  try {
    const response = await api.buildIndex(selectedSymbol.value);
    console.log("Index rebuilt:", response);
    await refreshAll();
  } catch (error) {
    console.error("Failed to rebuild index:", error);
  } finally {
    isLoading.value = false;
  }
};

// Watch symbol change
watch(selectedSymbol, () => {
  console.log("🔄 Symbol changed to:", selectedSymbol.value);
  refreshAll();
});

// Handle symbol change
const onSymbolChange = () => {
  console.log("📊 Selected symbol:", selectedSymbol.value);
};

// Lifecycle
onMounted(() => {
  refreshAll();
  checkSystemHealth();
  // Auto refresh every 10 seconds
  refreshInterval = setInterval(refreshAll, 10000);
});

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
});
</script>
