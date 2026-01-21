<template>
  <div class="space-y-4">
    <!-- Titan Core Details -->
    <div v-if="moduleName === 'titan'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Consensus Level</div>
          <div :class="getConsensusClass(data.consensus)">
            {{ data.consensus || "UNKNOWN" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Market Condition</div>
          <div :class="getMarketConditionClass(data.market_condition)">
            {{ data.market_condition || "UNKNOWN" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Prediction Direction</div>
          <div :class="getPredictionClass(data.prediction)">
            {{ data.prediction || "N/A" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Calibrated Confidence</div>
          <div class="text-white font-bold">
            {{ data.calibration?.toFixed(1) || 0 }}%
          </div>
        </div>
      </div>

      <!-- Module Weights -->
      <div v-if="data.module_weights" class="bg-gray-600 rounded-lg p-3">
        <div class="text-gray-400 text-xs mb-2">Module Weights</div>
        <div class="space-y-2">
          <div
            v-for="(weight, mod) in data.module_weights"
            :key="mod"
            class="flex items-center gap-2"
          >
            <span class="text-gray-300 text-sm w-20 truncate">{{ mod }}</span>
            <div class="flex-1 bg-gray-700 rounded-full h-2">
              <div
                class="h-2 rounded-full bg-blue-500"
                :style="{ width: weight * 100 + '%' }"
              ></div>
            </div>
            <span class="text-gray-400 text-xs w-12 text-right"
              >{{ (weight * 100).toFixed(0) }}%</span
            >
          </div>
        </div>
      </div>
    </div>

    <!-- Omega Brain Details -->
    <div v-else-if="moduleName === 'omega'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Institutional Flow</div>
          <div :class="getFlowClass(data.institutional_flow)">
            {{ data.institutional_flow || "NEUTRAL" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Manipulation Alert</div>
          <div :class="getManipulationClass(data.manipulation)">
            {{ data.manipulation || "NONE" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Sentiment Fusion</div>
          <div :class="getSentimentClass(data.sentiment)">
            {{ data.sentiment || "NEUTRAL" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Regime Prediction</div>
          <div class="text-white">{{ data.regime_prediction || "N/A" }}</div>
        </div>
      </div>

      <!-- Position Plan -->
      <div v-if="data.position_plan" class="bg-gray-600 rounded-lg p-3">
        <div class="text-gray-400 text-xs mb-2">Position Orchestration</div>
        <div class="grid grid-cols-3 gap-2 text-sm">
          <div>
            <span class="text-gray-400">Entry:</span>
            <span class="text-white ml-1">{{
              data.position_plan.entry_style
            }}</span>
          </div>
          <div>
            <span class="text-gray-400">Scale:</span>
            <span class="text-white ml-1">{{
              data.position_plan.scale_points
            }}</span>
          </div>
          <div>
            <span class="text-gray-400">Risk:</span>
            <span class="text-white ml-1"
              >{{ data.position_plan.risk_parity?.toFixed(1) }}%</span
            >
          </div>
        </div>
      </div>
    </div>

    <!-- Alpha Engine Details -->
    <div v-else-if="moduleName === 'alpha'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Order Flow Bias</div>
          <div :class="getFlowClass(data.order_flow_bias)">
            {{ data.order_flow_bias || "NEUTRAL" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Liquidity Zone</div>
          <div class="text-white">{{ data.liquidity_zone || "N/A" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Divergence</div>
          <div :class="getDivergenceClass(data.divergence)">
            {{ data.divergence || "NONE" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Momentum Wave</div>
          <div class="text-white">{{ data.momentum_wave || "N/A" }}</div>
        </div>
      </div>

      <!-- Risk Metrics -->
      <div v-if="data.risk_metrics" class="bg-gray-600 rounded-lg p-3">
        <div class="text-gray-400 text-xs mb-2">Risk Metrics</div>
        <div class="grid grid-cols-4 gap-2 text-sm">
          <div class="text-center">
            <div class="text-gray-400 text-xs">Sharpe</div>
            <div class="text-white font-bold">
              {{ data.risk_metrics.sharpe?.toFixed(2) || 0 }}
            </div>
          </div>
          <div class="text-center">
            <div class="text-gray-400 text-xs">Sortino</div>
            <div class="text-white font-bold">
              {{ data.risk_metrics.sortino?.toFixed(2) || 0 }}
            </div>
          </div>
          <div class="text-center">
            <div class="text-gray-400 text-xs">Calmar</div>
            <div class="text-white font-bold">
              {{ data.risk_metrics.calmar?.toFixed(2) || 0 }}
            </div>
          </div>
          <div class="text-center">
            <div class="text-gray-400 text-xs">Max DD</div>
            <div class="text-red-400 font-bold">
              {{ (data.risk_metrics.max_dd * 100)?.toFixed(1) || 0 }}%
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Quantum Strategy Details -->
    <div v-else-if="moduleName === 'quantum'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Microstructure</div>
          <div class="text-white">{{ data.microstructure || "N/A" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Volatility Regime</div>
          <div :class="getVolatilityClass(data.volatility_regime)">
            {{ data.volatility_regime || "N/A" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Fractal Pattern</div>
          <div class="text-purple-400">{{ data.fractal || "NONE" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Exit Strategy</div>
          <div class="text-white">{{ data.exit_strategy || "N/A" }}</div>
        </div>
      </div>
    </div>

    <!-- Deep Intelligence Details -->
    <div v-else-if="moduleName === 'deep'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Cross-Asset Signal</div>
          <div :class="getPredictionClass(data.cross_asset_signal)">
            {{ data.cross_asset_signal || "N/A" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Correlation Score</div>
          <div class="text-white font-bold">
            {{ data.correlation?.toFixed(2) || 0 }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Adaptive Parameters</div>
          <div class="text-white">{{ data.adaptive_status || "N/A" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Predictive Model</div>
          <div class="text-white">{{ data.predictive_model || "N/A" }}</div>
        </div>
      </div>
    </div>

    <!-- Neural Brain Details -->
    <div v-else-if="moduleName === 'neural'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Pattern DNA Score</div>
          <div class="text-orange-400 font-bold text-xl">
            {{ data.dna_score?.toFixed(0) || 0 }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Market State</div>
          <div :class="getMarketStateClass(data.market_state)">
            {{ data.market_state || "N/A" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Anomaly Detected</div>
          <div :class="data.anomaly ? 'text-red-400' : 'text-green-400'">
            {{ data.anomaly ? "⚠️ YES" : "✓ NO" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Risk Intelligence</div>
          <div class="text-white">{{ data.risk_intelligence || "N/A" }}</div>
        </div>
      </div>
    </div>

    <!-- Continuous Learning Details -->
    <div v-else-if="moduleName === 'learning'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Learning Cycles</div>
          <div class="text-blue-400 font-bold text-xl">
            {{ data.cycles || 0 }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Market Cycle</div>
          <div class="text-white">{{ data.market_cycle || "N/A" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Pattern Evolution</div>
          <div class="text-white">{{ data.pattern_evolution || "N/A" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Strategy Params</div>
          <div class="text-white">{{ data.strategy_params || "Default" }}</div>
        </div>
      </div>
    </div>

    <!-- Advanced Intelligence Details -->
    <div v-else-if="moduleName === 'advanced'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Market Regime</div>
          <div :class="getRegimeClass(data.regime)">
            {{ data.regime || "N/A" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">MTF Alignment</div>
          <div :class="getMtfClass(data.mtf_alignment)">
            {{ data.mtf_alignment || "N/A" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">S/R Level</div>
          <div class="text-white">{{ data.sr_level || "N/A" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Confluence Score</div>
          <div class="text-white font-bold">
            {{ data.confluence?.toFixed(0) || 0 }}%
          </div>
        </div>
      </div>
    </div>

    <!-- Smart Brain Details -->
    <div v-else-if="moduleName === 'smart'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Patterns Learned</div>
          <div class="text-yellow-400 font-bold text-xl">
            {{ data.patterns || 0 }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Journal Entries</div>
          <div class="text-white font-bold">
            {{ data.journal_entries || 0 }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Best Time</div>
          <div class="text-white">{{ data.best_time || "N/A" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Best Symbol</div>
          <div class="text-white">{{ data.best_symbol || "N/A" }}</div>
        </div>
      </div>
    </div>

    <!-- Pro Trading Details -->
    <div v-else-if="moduleName === 'pro'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Session</div>
          <div :class="getSessionClass(data.session)">
            {{ data.session || "N/A" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">News Impact</div>
          <div :class="getNewsClass(data.news_impact)">
            {{ data.news_impact || "NONE" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Correlation Filter</div>
          <div :class="data.correlation_ok ? 'text-green-400' : 'text-red-400'">
            {{ data.correlation_ok ? "✓ PASSED" : "✗ BLOCKED" }}
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Losing Streak</div>
          <div
            :class="data.losing_streak > 3 ? 'text-red-400' : 'text-green-400'"
          >
            {{ data.losing_streak || 0 }} trades
          </div>
        </div>
      </div>
    </div>

    <!-- Risk Guardian Details -->
    <div v-else-if="moduleName === 'risk'" class="space-y-4">
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Risk Level</div>
          <div :class="getRiskClass(data.level)">{{ data.level || "N/A" }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Daily P/L</div>
          <div :class="data.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'">
            {{ data.daily_pnl >= 0 ? "+" : ""
            }}{{ data.daily_pnl?.toFixed(2) || 0 }}%
          </div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Open Positions</div>
          <div class="text-white font-bold">{{ data.open_positions || 0 }}</div>
        </div>
        <div class="bg-gray-600 rounded-lg p-3">
          <div class="text-gray-400 text-xs mb-1">Can Trade</div>
          <div :class="data.can_trade ? 'text-green-400' : 'text-red-400'">
            {{ data.can_trade ? "✓ YES" : "✗ NO" }}
          </div>
        </div>
      </div>

      <!-- Risk Breakdown -->
      <div class="bg-gray-600 rounded-lg p-3">
        <div class="text-gray-400 text-xs mb-2">Risk Limits</div>
        <div class="space-y-2">
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Max Risk/Trade</span>
            <span class="text-white">{{ data.max_risk_per_trade || 2 }}%</span>
          </div>
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Max Daily Loss</span>
            <span class="text-white">{{ data.max_daily_loss || 5 }}%</span>
          </div>
          <div class="flex justify-between text-sm">
            <span class="text-gray-400">Max Positions</span>
            <span class="text-white">{{ data.max_positions || 3 }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Default (Unknown Module) -->
    <div v-else class="text-gray-400 text-center py-4">
      No details available for this module
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  moduleName: {
    type: String,
    required: true,
  },
  data: {
    type: Object,
    default: () => ({}),
  },
});

// Helper functions for styling
const getConsensusClass = (consensus) => {
  if (!consensus) return "text-gray-400";
  if (consensus === "UNANIMOUS") return "text-green-400 font-bold";
  if (consensus === "STRONG") return "text-blue-400 font-bold";
  if (consensus === "MODERATE") return "text-yellow-400";
  if (consensus === "WEAK") return "text-orange-400";
  return "text-red-400";
};

const getMarketConditionClass = (condition) => {
  if (!condition) return "text-gray-400";
  if (condition.includes("FAVORABLE")) return "text-green-400";
  if (condition.includes("UNFAVORABLE")) return "text-red-400";
  return "text-yellow-400";
};

const getPredictionClass = (pred) => {
  if (!pred) return "text-gray-400";
  if (pred === "BUY" || pred === "BULLISH" || pred === "UP")
    return "text-green-400 font-bold";
  if (pred === "SELL" || pred === "BEARISH" || pred === "DOWN")
    return "text-red-400 font-bold";
  return "text-yellow-400";
};

const getFlowClass = (flow) => {
  if (!flow) return "text-gray-400";
  if (flow === "ACCUMULATION" || flow === "BULLISH") return "text-green-400";
  if (flow === "DISTRIBUTION" || flow === "BEARISH") return "text-red-400";
  return "text-yellow-400";
};

const getManipulationClass = (manip) => {
  if (!manip || manip === "NONE") return "text-green-400";
  return "text-red-400 font-bold animate-pulse";
};

const getSentimentClass = (sentiment) => {
  if (!sentiment) return "text-gray-400";
  if (sentiment === "BULLISH" || sentiment === "POSITIVE")
    return "text-green-400";
  if (sentiment === "BEARISH" || sentiment === "NEGATIVE")
    return "text-red-400";
  return "text-yellow-400";
};

const getDivergenceClass = (div) => {
  if (!div || div === "NONE") return "text-gray-400";
  if (div.includes("BULLISH")) return "text-green-400";
  if (div.includes("BEARISH")) return "text-red-400";
  return "text-yellow-400";
};

const getVolatilityClass = (vol) => {
  if (!vol) return "text-gray-400";
  if (vol === "LOW") return "text-green-400";
  if (vol === "NORMAL") return "text-blue-400";
  if (vol === "HIGH") return "text-yellow-400";
  if (vol === "EXTREME") return "text-red-400 animate-pulse";
  return "text-gray-400";
};

const getMarketStateClass = (state) => {
  if (!state) return "text-gray-400";
  if (state === "TRENDING") return "text-green-400";
  if (state === "RANGING") return "text-yellow-400";
  if (state === "VOLATILE") return "text-orange-400";
  if (state === "REVERSAL") return "text-red-400";
  return "text-gray-400";
};

const getRegimeClass = (regime) => {
  if (!regime) return "text-gray-400";
  if (regime === "UPTREND") return "text-green-400";
  if (regime === "DOWNTREND") return "text-red-400";
  if (regime === "RANGING") return "text-yellow-400";
  return "text-gray-400";
};

const getMtfClass = (alignment) => {
  if (!alignment) return "text-gray-400";
  if (alignment === "ALIGNED" || alignment === "STRONG")
    return "text-green-400";
  if (alignment === "PARTIAL") return "text-yellow-400";
  if (alignment === "CONFLICT") return "text-red-400";
  return "text-gray-400";
};

const getSessionClass = (session) => {
  if (!session) return "text-gray-400";
  if (session === "LONDON") return "text-blue-400 font-bold";
  if (session === "NEW_YORK") return "text-green-400 font-bold";
  if (session === "TOKYO") return "text-red-400";
  if (session === "SYDNEY") return "text-purple-400";
  return "text-gray-400";
};

const getNewsClass = (impact) => {
  if (!impact || impact === "NONE") return "text-green-400";
  if (impact === "LOW") return "text-yellow-400";
  if (impact === "MEDIUM") return "text-orange-400";
  if (impact === "HIGH") return "text-red-400 animate-pulse";
  return "text-gray-400";
};

const getRiskClass = (level) => {
  if (!level) return "text-gray-400";
  if (level === "SAFE") return "text-green-400 font-bold";
  if (level === "CAUTION") return "text-yellow-400 font-bold";
  if (level === "WARNING") return "text-orange-400 font-bold";
  if (level === "DANGER" || level === "CRITICAL")
    return "text-red-400 font-bold animate-pulse";
  return "text-gray-400";
};
</script>
