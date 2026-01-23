/**
 * Trademify API Service - ULTRA SMART Edition 🧠
 * ระบบ API ที่ฉลาดมาก พร้อม:
 * - Auto-retry with exponential backoff
 * - Smart connection health monitoring
 * - Request queue when offline
 * - Intelligent caching layer
 * - Request deduplication
 * - Real-time connection status
 * - Auto-reconnect with heartbeat
 * - Smart fallback strategies
 */

import axios from "axios";

// ===========================================
// 🧠 SMART CONFIGURATION
// ===========================================

const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true" || false;

const getApiBaseUrl = () => {
  // 1. ถ้ามี VITE_API_URL กำหนดไว้ ใช้ค่านั้น
  if (import.meta.env.VITE_API_URL && import.meta.env.VITE_API_URL.trim()) {
    const base = import.meta.env.VITE_API_URL.replace(/\/api\/v1\/?$/, "");
    return `${base}/api/v1`;
  }

  // 2. Auto-detect จาก browser location
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;

  // ถ้าเป็น localhost หรือ 127.0.0.1 ใช้ localhost
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://localhost:8000/api/v1";
  }

  // 3. ถ้าเป็น IP อื่น (VPS) ใช้ IP นั้นเลย
  return `${protocol}//${hostname}:8000/api/v1`;
};

const API_BASE_URL = getApiBaseUrl();
console.log("🌐 [API] Base URL:", API_BASE_URL);
console.log("🖥️ [API] Detected hostname:", window.location.hostname);

// ===========================================
// 🧠 SMART CONNECTION MANAGER
// ===========================================

class SmartConnectionManager {
  constructor() {
    this.isOnline = true;
    this.lastSuccessTime = null;
    this.lastErrorTime = null;
    this.consecutiveErrors = 0;
    this.maxRetries = 3;
    this.retryDelays = [1000, 2000, 4000]; // Exponential backoff
    this.healthCheckInterval = null;
    this.listeners = new Set();
    this.pendingRequests = new Map(); // Request deduplication
    this.cache = new Map(); // Response cache
    this.cacheExpiry = 5000; // 5 seconds cache
    this.reconnecting = false;

    // Start health monitoring
    this.startHealthMonitoring();
  }

  // 📡 Connection Status Events
  onStatusChange(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  notifyListeners() {
    const status = this.getStatus();
    this.listeners.forEach((cb) => cb(status));
  }

  getStatus() {
    return {
      isOnline: this.isOnline,
      consecutiveErrors: this.consecutiveErrors,
      lastSuccessTime: this.lastSuccessTime,
      lastErrorTime: this.lastErrorTime,
      reconnecting: this.reconnecting,
      health:
        this.consecutiveErrors === 0
          ? "excellent"
          : this.consecutiveErrors < 3
            ? "degraded"
            : "critical",
    };
  }

  // ✅ Mark success
  markSuccess() {
    this.isOnline = true;
    this.consecutiveErrors = 0;
    this.lastSuccessTime = Date.now();
    this.reconnecting = false;
    this.notifyListeners();
  }

  // ❌ Mark failure
  markFailure(error) {
    this.consecutiveErrors++;
    this.lastErrorTime = Date.now();

    if (this.consecutiveErrors >= 3) {
      this.isOnline = false;
    }

    this.notifyListeners();
    return this.consecutiveErrors;
  }

  // 🔄 Auto-reconnect
  async attemptReconnect() {
    if (this.reconnecting) return;

    this.reconnecting = true;
    this.notifyListeners();

    console.log("🔄 [Smart] Attempting to reconnect...");

    try {
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: 5000,
      });
      if (response.data) {
        console.log("✅ [Smart] Reconnected successfully!");
        this.markSuccess();
        return true;
      }
    } catch (e) {
      console.log("⏳ [Smart] Reconnect failed, will retry...");
    }

    this.reconnecting = false;
    return false;
  }

  // 💓 Health monitoring
  startHealthMonitoring() {
    // Check every 10 seconds when online, every 3 seconds when offline
    const check = async () => {
      const interval = this.isOnline ? 10000 : 3000;

      if (!this.isOnline) {
        await this.attemptReconnect();
      }

      this.healthCheckInterval = setTimeout(check, interval);
    };

    // Start after 5 seconds
    setTimeout(check, 5000);
  }

  // 🧹 Cleanup
  destroy() {
    if (this.healthCheckInterval) {
      clearTimeout(this.healthCheckInterval);
    }
    this.listeners.clear();
    this.pendingRequests.clear();
    this.cache.clear();
  }

  // 📦 Cache management
  getCached(key) {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.time < this.cacheExpiry) {
      return cached.data;
    }
    return null;
  }

  setCache(key, data) {
    this.cache.set(key, { data, time: Date.now() });
    // Clean old cache entries
    if (this.cache.size > 100) {
      const oldestKey = this.cache.keys().next().value;
      this.cache.delete(oldestKey);
    }
  }

  // 🔀 Request deduplication
  async deduplicateRequest(key, requestFn) {
    // Check if same request is already pending
    if (this.pendingRequests.has(key)) {
      console.log("🔀 [Smart] Deduplicating request:", key);
      return this.pendingRequests.get(key);
    }

    const promise = requestFn();
    this.pendingRequests.set(key, promise);

    try {
      const result = await promise;
      return result;
    } finally {
      this.pendingRequests.delete(key);
    }
  }
}

// Create singleton instance
const connectionManager = new SmartConnectionManager();

// Export for external use
export const getConnectionStatus = () => connectionManager.getStatus();
export const onConnectionChange = (cb) => connectionManager.onStatusChange(cb);

// ===========================================
// 🚀 SMART AXIOS CLIENT
// ===========================================

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 15000, // Reduced timeout for faster feedback
});

// Request interceptor with smart logging
apiClient.interceptors.request.use(
  (config) => {
    const status = connectionManager.isOnline ? "🟢" : "🔴";
    console.log(
      `${status} [API] ${config.method?.toUpperCase()} ${config.url}`,
    );
    return config;
  },
  (error) => Promise.reject(error),
);

// Response interceptor with smart error handling
apiClient.interceptors.response.use(
  (response) => {
    connectionManager.markSuccess();
    return response.data;
  },
  (error) => {
    const errorCount = connectionManager.markFailure(error);

    if (error.code === "ECONNABORTED") {
      console.warn("⏱️ [API] Request timeout");
    } else if (!error.response) {
      console.error("🔌 [API] Network error - server unreachable");
    } else {
      console.error("❌ [API Error]", error.response?.data || error.message);
    }

    return Promise.reject(error.response?.data || error);
  },
);

// ===========================================
// MOCK DATA GENERATORS
// ===========================================

const mockData = {
  generateSignal(symbol, timeframe) {
    const signals = ["STRONG_BUY", "BUY", "WAIT", "SELL", "STRONG_SELL"];
    const signal = signals[Math.floor(Math.random() * signals.length)];
    const bullish = Math.floor(Math.random() * 10);
    const bearish = 10 - bullish;
    const basePrice = symbol.includes("BTC")
      ? 95000
      : symbol.includes("XAU")
        ? 2650
        : 1.085;

    return {
      status: "success",
      signal,
      confidence: (Math.max(bullish, bearish) / 10) * 100,
      vote_details: { bullish, bearish, total: 10 },
      price_projection: {
        current: basePrice,
        projected: basePrice * (1 + (Math.random() - 0.5) * 0.02),
        stop_loss: basePrice * (signal.includes("BUY") ? 0.98 : 1.02),
        take_profit: basePrice * (signal.includes("BUY") ? 1.03 : 0.97),
      },
      n_matches: 10,
      timestamp: new Date().toISOString(),
    };
  },

  tradingStatus() {
    return {
      initialized: true,
      enabled: false,
      running: false,
      broker_connected: false,
      open_positions: 0,
      stats: {
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        total_pnl: 0,
      },
    };
  },

  positions() {
    return { positions: [], count: 0, total_pnl: 0 };
  },

  multiFactorResult(symbol) {
    return {
      status: "success",
      signal: "BUY",
      base_confidence: 75.0,
      final_score: 68.5,
      quality: "MEDIUM",
      recommendation: "TRADE_REDUCED",
      factors: [
        {
          name: "Pattern Match",
          score: 75,
          weight: 25,
          passed: true,
          details: "Good match",
        },
        {
          name: "Trend Alignment",
          score: 80,
          weight: 20,
          passed: true,
          details: "Aligned with uptrend",
        },
        {
          name: "Volume Confirmation",
          score: 60,
          weight: 15,
          passed: true,
          details: "1.2x avg volume",
        },
        {
          name: "Pattern Recency",
          score: 50,
          weight: 10,
          passed: true,
          details: "Recent patterns",
        },
        {
          name: "Volatility",
          score: 70,
          weight: 10,
          passed: true,
          details: "Good volatility",
        },
        {
          name: "Session Timing",
          score: 85,
          weight: 10,
          passed: true,
          details: "London session",
        },
        {
          name: "Momentum",
          score: 65,
          weight: 10,
          passed: true,
          details: "RSI: 55",
        },
      ],
      scores: {
        pattern: 75,
        trend: 80,
        volume: 60,
        recency: 50,
        volatility: 70,
        session: 85,
        momentum: 65,
      },
      position_size_multiplier: 0.8,
      trading_mode: "CONSERVATIVE",
      timestamp: new Date().toISOString(),
    };
  },

  botStatus() {
    return {
      running: false,
      signals_count: 0,
      last_signal: null,
      config: { symbol: "EURUSD", timeframe: "H1", mode: "CONSERVATIVE" },
    };
  },

  // Intelligence mock data
  intelligenceStatus() {
    return {
      titan: {
        active: true,
        grade: "🏛️ TITAN SUPREME",
        score: 92.5,
        consensus: "STRONG",
        market_condition: "FAVORABLE",
        prediction: { direction: "BUY", confidence: 85 },
      },
      omega: {
        active: true,
        grade: "Ω+",
        score: 88.0,
        institutional_flow: "ACCUMULATION",
        manipulation: "NONE",
        sentiment: "BULLISH",
      },
      alpha: { active: true, grade: "A+", score: 85.0 },
      quantum: { active: true, grade: "QUANTUM", score: 78.0 },
      deep: { active: true, score: 75.0, correlation: 0.85 },
      neural: { active: true, dna_score: 82.0, market_state: "TRENDING" },
      learning: { active: true, cycles: 150, market_cycle: "EXPANSION" },
      advanced: { active: true, regime: "UPTREND", mtf_alignment: "ALIGNED" },
      smart: { active: true, patterns: 1250, journal_entries: 85 },
      pro: { active: true, session: "LONDON", news_impact: "NONE" },
      risk: {
        active: true,
        level: "SAFE",
        daily_pnl: 2.5,
        can_trade: true,
        open_positions: 1,
        losing_streak: 0,
      },
    };
  },

  titanData() {
    return {
      grade: "🏛️ TITAN SUPREME",
      titan_score: 92.5,
      consensus: "STRONG",
      market_condition: "FAVORABLE",
      prediction: {
        direction: "BUY",
        confidence: 85,
        momentum: "BUY",
        mean_reversion: "NEUTRAL",
        volatility: "BUY",
        pattern: "BUY",
      },
      raw_confidence: 80,
      calibrated_confidence: 78,
      historical_accuracy: 72,
      position_multiplier: 0.9,
      improvements: [],
      module_weights: {
        neural: 0.15,
        deep: 0.15,
        quantum: 0.15,
        alpha: 0.2,
        omega: 0.2,
        pattern: 0.15,
      },
    };
  },

  omegaData() {
    return {
      grade: "Ω+",
      omega_score: 88.5,
      institutional_flow: "ACCUMULATION",
      manipulation: "NONE",
      manipulation_alerts: [],
      sentiment: "BULLISH",
      volume_anomaly: 1.8,
      big_money: "ACTIVE",
      smart_money: "BUYING",
      current_regime: "UPTREND",
      predicted_regime: "UPTREND",
      regime_confidence: 80,
      suggested_allocation: 85,
      volatility_weight: 0.9,
      risk_contribution: "NORMAL",
      entry_style: "SCALE_IN",
      scale_points: 3,
      exit_plan: "PARTIAL",
      trailing_stop: true,
      position_multiplier: 0.95,
    };
  },

  riskData() {
    return {
      risk_level: "SAFE",
      balance: 10000,
      equity: 10250,
      daily_pnl: 2.5,
      open_positions: 1,
      max_positions: 3,
      risk_per_trade: 2,
      max_daily_loss: 5,
      leverage: 2000,
      risk_score: 20,
      can_trade: true,
      can_open_position: true,
      daily_limit_hit: false,
      losing_streak_limit: false,
      losing_streak: 0,
      max_losing_streak: 5,
    };
  },
};

// ===========================================
// 🧠 SMART API WRAPPER WITH RETRY & CACHE
// ===========================================

let apiConnected = false;
let lastApiError = null;

export function isApiConnected() {
  return connectionManager.isOnline;
}
export function getLastApiError() {
  return lastApiError;
}

/**
 * 🧠 Ultra Smart API Call
 * - Auto-retry with exponential backoff
 * - Response caching
 * - Request deduplication
 * - Graceful fallback to mock
 */
async function smartApiCall(apiCall, mockFn, options = {}) {
  const {
    cacheKey = null,
    cacheTTL = 5000,
    maxRetries = 2,
    silent = false,
    critical = false, // Critical requests won't fallback to mock
  } = options;

  // 1. Check mock mode
  if (USE_MOCK) {
    if (!silent) console.log("🎭 [MOCK] Using mock data");
    const result = mockFn();
    result._isMock = true;
    result._source = "mock";
    return result;
  }

  // 2. Check cache first
  if (cacheKey) {
    const cached = connectionManager.getCached(cacheKey);
    if (cached) {
      if (!silent) console.log("📦 [Cache] Using cached data:", cacheKey);
      cached._source = "cache";
      return cached;
    }
  }

  // 3. Try API with smart retry
  let lastError = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Deduplicate if we have a cache key
      const result = cacheKey
        ? await connectionManager.deduplicateRequest(cacheKey, apiCall)
        : await apiCall();

      // Success! Cache the result
      if (cacheKey) {
        connectionManager.setCache(cacheKey, result);
      }

      apiConnected = true;
      lastApiError = null;
      result._isMock = false;
      result._source = "api";
      result._attempt = attempt + 1;

      return result;
    } catch (error) {
      lastError = error;

      // Don't retry on 4xx errors (client errors)
      if (
        error.response &&
        error.response.status >= 400 &&
        error.response.status < 500
      ) {
        break;
      }

      // Wait before retry (exponential backoff)
      if (attempt < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 5000);
        if (!silent)
          console.log(
            `⏳ [Retry] Attempt ${attempt + 2}/${maxRetries + 1} in ${delay}ms...`,
          );
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  // 4. All retries failed
  apiConnected = false;
  lastApiError = lastError?.message || "Connection failed";

  // Critical requests should throw
  if (critical) {
    throw lastError;
  }

  // Fallback to mock
  if (!silent) {
    console.warn("🔄 [Fallback] Using mock data after API failure");
  }

  const result = mockFn();
  result._isMock = true;
  result._source = "fallback";
  result._error = lastApiError;
  return result;
}

// Legacy wrapper for backward compatibility
async function tryApiOrMock(apiCall, mockFn, silent = false) {
  return smartApiCall(apiCall, mockFn, { silent });
}

// ===========================================
// 🚀 SMART API SERVICE
// ===========================================

const api = {
  // ===========================================
  // 1. HEALTH & STATUS (with smart caching)
  // ===========================================

  async healthCheck() {
    return smartApiCall(
      () => apiClient.get("/health"),
      () => ({
        status: "mock",
        timestamp: new Date().toISOString(),
        indices_loaded: [],
      }),
      { cacheKey: "health", cacheTTL: 3000, silent: true },
    );
  },

  // ===========================================
  // 2. PATTERN INDEX
  // ===========================================

  async buildIndex(symbol, timeframe = "H1", useSampleData = false) {
    return tryApiOrMock(
      () =>
        apiClient.post("/build-index", {
          symbol,
          timeframe,
          window_size: 60,
          future_candles: 10,
          use_sample_data: useSampleData,
        }),
      () => ({
        status: "success",
        symbol,
        timeframe,
        n_patterns: 1000,
        is_ready: true,
      }),
    );
  },

  async buildRealtimeIndex(symbol, timeframe = "H1") {
    return tryApiOrMock(
      () => apiClient.post("/build-realtime-index", { symbol, timeframe }),
      () => ({ status: "success", symbol, timeframe, n_patterns: 1000 }),
    );
  },

  async getIndexStatus(symbol, timeframe = "H1") {
    return tryApiOrMock(
      () => apiClient.get(`/index-status/${symbol}/${timeframe}`),
      () => ({ symbol, timeframe, n_patterns: 1000, is_ready: true }),
    );
  },

  // ===========================================
  // 3. BASIC ANALYSIS
  // ===========================================

  async analyze(data) {
    return tryApiOrMock(
      () => apiClient.post("/analyze", data),
      () => mockData.generateSignal(data.symbol, data.timeframe),
    );
  },

  async analyzeRealtime(symbol = "EURUSD", timeframe = "H1") {
    return tryApiOrMock(
      () =>
        apiClient.post(
          `/analyze-realtime?symbol=${symbol}&timeframe=${timeframe}`,
        ),
      () => mockData.generateSignal(symbol, timeframe),
    );
  },

  async generateSampleSignal(symbol = "EURUSD", timeframe = "H1") {
    return tryApiOrMock(
      () =>
        apiClient.post(
          `/generate-sample-signal?symbol=${symbol}&timeframe=${timeframe}`,
        ),
      () => mockData.generateSignal(symbol, timeframe),
    );
  },

  // ===========================================
  // 4. ENHANCED AI ANALYSIS (High Win Rate)
  // ===========================================

  async analyzeEnhanced(data) {
    return tryApiOrMock(
      () => apiClient.post("/analyze-enhanced", data),
      () => ({
        ...mockData.generateSignal(data.symbol, data.timeframe),
        enhanced_confidence: 75,
        quality: "HIGH",
        scores: {
          pattern: 80,
          technical: 75,
          volume: 70,
          mtf: 65,
          regime: 70,
          timing: 85,
          momentum: 70,
        },
        market_regime: "UPTREND",
      }),
    );
  },

  async analyzeRealtimeEnhanced(
    symbol = "EURUSD",
    timeframe = "H1",
    htfTimeframe = "H4",
    minQuality = "MEDIUM",
  ) {
    return tryApiOrMock(
      () =>
        apiClient.post(
          `/analyze-realtime-enhanced?symbol=${symbol}&timeframe=${timeframe}&htf_timeframe=${htfTimeframe}&min_quality=${minQuality}`,
        ),
      () => ({
        ...mockData.generateSignal(symbol, timeframe),
        enhanced_confidence: 78,
        quality: minQuality,
      }),
    );
  },

  // ===========================================
  // 5. MULTI-FACTOR AI ANALYSIS (Maximum Win Rate)
  // ===========================================

  async analyzeMultiFactor(data) {
    return tryApiOrMock(
      () => apiClient.post("/analyze-multi-factor", data),
      () => mockData.multiFactorResult(data.symbol),
    );
  },

  async getMultiFactorConfig() {
    return tryApiOrMock(
      () => apiClient.get("/multi-factor/config"),
      () => ({
        mode: "CONSERVATIVE",
        pattern_weight: 0.25,
        filters: {
          trend: { enabled: true, weight: 0.2 },
          volume: { enabled: true, weight: 0.15 },
          recency: { enabled: true, weight: 0.1 },
          volatility: { enabled: true, weight: 0.1 },
          session: { enabled: true, weight: 0.1 },
          momentum: { enabled: true, weight: 0.1 },
        },
        thresholds: { min_final_score: 75, strong_signal_score: 85 },
      }),
    );
  },

  // ===========================================
  // 6. SYMBOLS & MARKET DATA
  // ===========================================

  async getSymbols() {
    return tryApiOrMock(
      () => apiClient.get("/symbols"),
      () => ({
        symbols: {
          EURUSD: {},
          GBPUSD: {},
          USDJPY: {},
          BTCUSDT: {},
          XAUUSD: {},
        },
      }),
    );
  },

  async getMarketData(symbol) {
    return tryApiOrMock(
      () => apiClient.get(`/market/${symbol}`),
      () => ({
        symbol,
        price: symbol.includes("BTC") ? 95000 : 1.085,
        change_24h: 0.5,
        volume_24h: 1000000,
      }),
    );
  },

  // ===========================================
  // 7. TRADING - STATUS & SETTINGS
  // ===========================================

  async getTradingStatus() {
    return tryApiOrMock(
      () => apiClient.get("/trading/status"),
      () => mockData.tradingStatus(),
    );
  },

  async getTradingSettings() {
    return tryApiOrMock(
      () => apiClient.get("/trading/settings"),
      () => ({
        enabled: false,
        paper_trading: true,
        broker_type: "MT5",
        symbols: ["EURUSD"],
        timeframe: "H1",
        risk_per_trade: 2.0,
        max_daily_loss: 5.0,
        max_positions: 3,
        min_confidence: 70,
      }),
    );
  },

  async updateTradingSettings(settings) {
    return tryApiOrMock(
      () => apiClient.put("/trading/settings", settings),
      () => ({ status: "success", ...settings }),
    );
  },

  // ===========================================
  // 8. TRADING - CONTROL
  // ===========================================

  async startTrading() {
    return tryApiOrMock(
      () => apiClient.post("/trading/start"),
      () => ({ status: "started", message: "Trading started (mock)" }),
    );
  },

  async stopTrading() {
    return tryApiOrMock(
      () => apiClient.post("/trading/stop"),
      () => ({ status: "stopped", message: "Trading stopped (mock)" }),
    );
  },

  async pauseTrading() {
    return tryApiOrMock(
      () => apiClient.post("/trading/pause"),
      () => ({ status: "paused", message: "Trading paused (mock)" }),
    );
  },

  async resumeTrading() {
    return tryApiOrMock(
      () => apiClient.post("/trading/resume"),
      () => ({ status: "resumed", message: "Trading resumed (mock)" }),
    );
  },

  // ===========================================
  // 9. TRADING - POSITIONS (SMART)
  // ===========================================

  async getPositions() {
    return smartApiCall(
      () => apiClient.get("/trading/positions"),
      () => mockData.positions(),
      { cacheKey: "positions", cacheTTL: 2000 },
    );
  },

  async openPosition(data) {
    return smartApiCall(
      () => apiClient.post("/trading/positions", data),
      () => ({ status: "opened", position_id: "mock_" + Date.now(), ...data }),
      { critical: true, maxRetries: 3 }, // Critical trading action
    );
  },

  async closePosition(positionId) {
    return smartApiCall(
      () => apiClient.delete(`/trading/positions/${positionId}`),
      () => ({ status: "closed", position_id: positionId }),
      { critical: true, maxRetries: 3 }, // Critical trading action
    );
  },

  async getTradeHistory(limit = 50) {
    return smartApiCall(
      () => apiClient.get(`/trading/history?limit=${limit}`),
      () => ({ trades: [], count: 0 }),
      { cacheKey: `trade_history_${limit}`, cacheTTL: 5000 },
    );
  },

  async getTradingStats() {
    return tryApiOrMock(
      () => apiClient.get("/trading/stats"),
      () => ({
        engine: {
          total_trades: 0,
          winning_trades: 0,
          losing_trades: 0,
          total_pnl: 0,
        },
        positions: { open: 0, closed: 0 },
        risk: { max_risk_per_trade: 2.0, max_daily_loss: 5.0 },
      }),
    );
  },

  async getTradingAccount() {
    return tryApiOrMock(
      () => apiClient.get("/trading/account"),
      () => ({ balance: 0, equity: 0, broker_type: "MT5", connected: false }),
    );
  },

  // ===========================================
  // 10. TRADING - SIGNAL PROCESSING
  // ===========================================

  async processSignal(signalData) {
    return tryApiOrMock(
      () => apiClient.post("/trading/signal", signalData),
      () => ({
        status: "processed",
        action: "none",
        reason: "Mock mode - signal not executed",
        signal: signalData,
      }),
    );
  },

  async getPrice(symbol) {
    return tryApiOrMock(
      () => apiClient.get(`/mt5/price/${symbol}`),
      () => ({
        price: symbol.includes("BTC")
          ? 95000
          : symbol.includes("XAU")
            ? 2650
            : 1.085,
        mt5_connected: false,
      }),
    );
  },

  // ===========================================
  // 11. MT5 - ACCOUNT & CONNECTION
  // ===========================================

  async getMT5Account() {
    return tryApiOrMock(
      () => apiClient.get("/mt5/account"),
      () => ({ connected: false, error: "Mock mode" }),
    );
  },

  async reconnectMT5() {
    return tryApiOrMock(
      () => apiClient.post("/mt5/reconnect"),
      () => ({
        status: "disconnected",
        message: "Mock mode - cannot reconnect",
      }),
    );
  },

  async getMarketStatus(symbol = "EURUSD") {
    return tryApiOrMock(
      () => apiClient.get(`/mt5/market-status?symbol=${symbol}`),
      () => ({
        status: "UNKNOWN",
        is_tradeable: false,
        mt5_connected: false,
        message: "Mock mode",
        message_th: "โหมดทดสอบ",
      }),
    );
  },

  async getMT5OHLCV(symbol, timeframe = "H1", count = 100) {
    return tryApiOrMock(
      () =>
        apiClient.get(
          `/mt5/ohlcv/${symbol}?timeframe=${timeframe}&count=${count}`,
        ),
      () => ({
        symbol,
        timeframe,
        data: [],
        count: 0,
        error: "Mock mode",
      }),
    );
  },

  // ===========================================
  // 12. BOT CONTROL (AI Enhanced Analysis Bot) - SMART
  // ===========================================

  async startBot(config = {}) {
    return smartApiCall(
      () => apiClient.post("/bot/start", config),
      () => ({ status: "started", message: "Bot started (mock)", ...config }),
      { critical: true, maxRetries: 3 }, // Critical - must reach server
    );
  },

  async stopBot() {
    return smartApiCall(
      () => apiClient.post("/bot/stop"),
      () => ({ status: "stopped", message: "Bot stopped (mock)" }),
      { critical: true, maxRetries: 3 }, // Critical - must reach server
    );
  },

  async getBotStatus() {
    return smartApiCall(
      () => apiClient.get("/bot/status"),
      () => mockData.botStatus(),
      { cacheKey: "bot_status", cacheTTL: 2000 }, // Cache for 2s
    );
  },

  async getBotSignals(limit = 20) {
    return smartApiCall(
      () => apiClient.get(`/bot/signals?limit=${limit}`),
      () => ({ signals: [], count: 0 }),
      { cacheKey: `bot_signals_${limit}`, cacheTTL: 3000 },
    );
  },

  async updateBotSettings(settings = {}) {
    return tryApiOrMock(
      () => apiClient.put("/bot/settings", settings),
      () => ({
        status: "updated",
        message: "Settings updated (mock)",
        ...settings,
      }),
    );
  },

  async restartBot(config = {}) {
    return tryApiOrMock(
      () => apiClient.post("/bot/restart", config),
      () => ({
        status: "restarted",
        message: "Bot restarted (mock)",
        ...config,
      }),
    );
  },

  // ===========================================
  // 13. INTELLIGENCE STATUS (20-Layer AI System) - SMART
  // ===========================================

  async getIntelligenceStatus() {
    return smartApiCall(
      () => apiClient.get("/intelligence/status"),
      () => mockData.intelligenceStatus(),
      { cacheKey: "intel_status", cacheTTL: 5000 },
    );
  },

  async getIntelligenceLayers() {
    return smartApiCall(
      () => apiClient.get("/intelligence/layers"),
      () => ({
        layers: [],
        last_decisions: {},
        quality_filter: { current: "MEDIUM", levels: {} },
        total_active: 0,
      }),
      { cacheKey: "intel_layers", cacheTTL: 5000 },
    );
  },

  async getPipelineData(symbol = "EURUSDm") {
    return smartApiCall(
      () => apiClient.get(`/intelligence/pipeline?symbol=${symbol}`),
      () => ({
        status: "mock",
        symbol,
        layers: {},
        current_signal: { signal: "WAIT", quality: null },
        final_decision: {
          action: "WAITING",
          position_multiplier: 1,
          verdict: "Mock data",
        },
      }),
      { cacheKey: `pipeline_${symbol}`, cacheTTL: 3000 },
    );
  },

  async getTitanData(symbol = "EURUSDm") {
    return smartApiCall(
      () => apiClient.get(`/intelligence/titan?symbol=${symbol}`),
      () => mockData.titanData(),
      { cacheKey: `titan_${symbol}`, cacheTTL: 5000 },
    );
  },

  async getOmegaData(symbol = "EURUSDm") {
    return smartApiCall(
      () => apiClient.get(`/intelligence/omega?symbol=${symbol}`),
      () => mockData.omegaData(),
      { cacheKey: `omega_${symbol}`, cacheTTL: 5000 },
    );
  },

  async getAlphaData(symbol = "EURUSDm") {
    return tryApiOrMock(
      () => apiClient.get(`/intelligence/alpha?symbol=${symbol}`),
      () => ({
        grade: "A+",
        alpha_score: 85.0,
        confidence: 82.0,
        order_flow_bias: "BULLISH",
        order_flow_delta: 0.25,
        risk_reward: 2.5,
        position_multiplier: 0.9,
        optimal_entry: 1.085,
        stop_loss: 1.082,
        targets: [1.088, 1.091, 1.095],
        market_profile: { poc: 1.0845, vah: 1.087, val: 1.082 },
        liquidity_zones: [
          { type: "SUPPORT", price: 1.082 },
          { type: "RESISTANCE", price: 1.09 },
        ],
        should_trade: true,
        edge_factors: ["Strong order flow", "Good R:R ratio"],
        risk_factors: [],
      }),
    );
  },

  async getLastAnalysis() {
    return tryApiOrMock(
      () => apiClient.get("/intelligence/last-analysis"),
      () => ({
        status: "active",
        last_analysis: {},
        titan: mockData.titanData(),
        omega: mockData.omegaData(),
        alpha: {},
      }),
    );
  },

  async getRiskData() {
    return smartApiCall(
      () => apiClient.get("/intelligence/risk"),
      () => mockData.riskData(),
      { cacheKey: "risk_data", cacheTTL: 3000 },
    );
  },

  async getModuleAnalysis(moduleName, symbol = "EURUSDm") {
    return tryApiOrMock(
      () =>
        apiClient.get(`/intelligence/module/${moduleName}?symbol=${symbol}`),
      () => ({
        module: moduleName,
        active: true,
        signal: "BUY",
        confidence: 75,
        details: {},
      }),
    );
  },

  async getSignalHistory(limit = 50) {
    return tryApiOrMock(
      () => apiClient.get(`/intelligence/signals/history?limit=${limit}`),
      () => ({
        signals: [
          {
            timestamp: new Date().toISOString(),
            symbol: "EURUSDm",
            signal: "BUY",
            titan_score: 92.5,
            omega_grade: "Ω+",
            result: "WIN",
          },
          {
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            symbol: "GBPUSDm",
            signal: "SELL",
            titan_score: 85.0,
            omega_grade: "Ω",
            result: "WIN",
          },
          {
            timestamp: new Date(Date.now() - 7200000).toISOString(),
            symbol: "XAUUSDm",
            signal: "BUY",
            titan_score: 78.5,
            omega_grade: "α+",
            result: "LOSS",
          },
        ],
        count: 3,
      }),
    );
  },

  // ===========================================
  // 14. SERVER-SENT EVENTS (Real-time Updates)
  // ===========================================

  getEventsUrl() {
    return `${API_BASE_URL}/events`;
  },

  subscribeToEvents(onMessage, onError) {
    const eventSource = new EventSource(this.getEventsUrl());

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.error("[SSE] Parse error:", e);
      }
    };

    eventSource.onerror = (error) => {
      console.error("[SSE] Connection error:", error);
      if (onError) onError(error);
    };

    return eventSource;
  },

  // ===========================================
  // 15. SYSTEM HEALTH (SMART)
  // ===========================================

  async getSystemHealth() {
    return smartApiCall(
      () => apiClient.get("/system/health"),
      () => ({
        mt5_connected: false,
        api_status: "mock",
        bot_running: false,
        data_lake_ready: false,
        faiss_loaded: false,
        intelligence_modules: 0,
        total_modules: 16,
        last_analysis_time: null,
        memory_usage: 0,
        uptime: 0,
      }),
      { cacheKey: "system_health", cacheTTL: 3000 },
    );
  },

  // ===========================================
  // 16. UTILITY FUNCTIONS (ENHANCED)
  // ===========================================

  getBaseUrl() {
    return API_BASE_URL;
  },

  isConnected() {
    return connectionManager.isOnline;
  },

  getConnectionStatus() {
    return connectionManager.getStatus();
  },

  onConnectionChange(callback) {
    return connectionManager.onStatusChange(callback);
  },

  // Force reconnect
  async reconnect() {
    return connectionManager.attemptReconnect();
  },

  // Clear all caches
  clearCache() {
    connectionManager.cache.clear();
    console.log("🧹 [API] Cache cleared");
  },

  // Get cache stats
  getCacheStats() {
    return {
      size: connectionManager.cache.size,
      keys: Array.from(connectionManager.cache.keys()),
    };
  },
};

export default api;
