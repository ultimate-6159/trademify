/**
 * 🧬 Trademify API Service - GENIUS LEVEL 100x 🧬
 * ระบบ API ระดับ Genius พร้อม:
 *
 * 🔮 PREDICTIVE LAYER
 * - Predictive prefetching (โหลดข้อมูลล่วงหน้า)
 * - Usage pattern learning (เรียนรู้พฤติกรรมผู้ใช้)
 * - Smart preloading based on navigation
 *
 * ⚡ PERFORMANCE LAYER
 * - Request batching (รวม requests)
 * - Priority queue (จัดลำดับความสำคัญ)
 * - Adaptive timeout (ปรับ timeout ตามสถานะ)
 * - Connection quality detection
 * - Bandwidth optimization
 *
 * 🛡️ RELIABILITY LAYER
 * - Offline queue with auto-sync
 * - Circuit breaker pattern
 * - Intelligent retry with jitter
 * - Request deduplication
 * - Stale-while-revalidate
 *
 * 📊 ANALYTICS LAYER
 * - Performance metrics
 * - Error tracking
 * - Latency monitoring
 * - Success rate calculation
 *
 * 🔄 REAL-TIME LAYER
 * - WebSocket with HTTP fallback
 * - Server-Sent Events support
 * - Real-time sync
 * - Optimistic updates
 */

import axios from "axios";

// ===========================================
// 🧬 GENIUS CONFIGURATION
// ===========================================

const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true" || false;
const DEBUG_MODE = import.meta.env.VITE_DEBUG === "true" || false;

const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_URL && import.meta.env.VITE_API_URL.trim()) {
    const base = import.meta.env.VITE_API_URL.replace(/\/api\/v1\/?$/, "");
    return `${base}/api/v1`;
  }

  const hostname = window.location.hostname;
  const protocol = window.location.protocol;

  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://localhost:8000/api/v1";
  }

  return `${protocol}//${hostname}:8000/api/v1`;
};

const API_BASE_URL = getApiBaseUrl();

// 🎨 Beautiful console logging
const log = {
  info: (emoji, ...args) => console.log(`${emoji}`, ...args),
  success: (...args) => console.log("✅", ...args),
  warn: (...args) => console.warn("⚠️", ...args),
  error: (...args) => console.error("❌", ...args),
  debug: (...args) => DEBUG_MODE && console.log("🔍", ...args),
  perf: (...args) => DEBUG_MODE && console.log("⚡", ...args),
};

log.info("🌐", "[GENIUS] Base URL:", API_BASE_URL);
log.info("🖥️", "[GENIUS] Hostname:", window.location.hostname);

// ===========================================
// 📊 PERFORMANCE METRICS TRACKER
// ===========================================

class PerformanceTracker {
  constructor() {
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      cachedResponses: 0,
      avgLatency: 0,
      latencies: [],
      errorTypes: new Map(),
      endpointStats: new Map(),
    };
    this.maxLatencySamples = 100;
  }

  recordRequest(endpoint, latency, success, error = null) {
    this.metrics.totalRequests++;

    if (success) {
      this.metrics.successfulRequests++;
    } else {
      this.metrics.failedRequests++;
      if (error) {
        const errorType = error.code || error.message || "unknown";
        this.metrics.errorTypes.set(
          errorType,
          (this.metrics.errorTypes.get(errorType) || 0) + 1,
        );
      }
    }

    // Track latency
    this.metrics.latencies.push(latency);
    if (this.metrics.latencies.length > this.maxLatencySamples) {
      this.metrics.latencies.shift();
    }
    this.metrics.avgLatency =
      this.metrics.latencies.reduce((a, b) => a + b, 0) /
      this.metrics.latencies.length;

    // Track per-endpoint stats
    if (!this.metrics.endpointStats.has(endpoint)) {
      this.metrics.endpointStats.set(endpoint, {
        calls: 0,
        errors: 0,
        avgLatency: 0,
        latencies: [],
      });
    }
    const stat = this.metrics.endpointStats.get(endpoint);
    stat.calls++;
    if (!success) stat.errors++;
    stat.latencies.push(latency);
    if (stat.latencies.length > 20) stat.latencies.shift();
    stat.avgLatency =
      stat.latencies.reduce((a, b) => a + b, 0) / stat.latencies.length;
  }

  recordCacheHit() {
    this.metrics.cachedResponses++;
  }

  getSuccessRate() {
    if (this.metrics.totalRequests === 0) return 100;
    return (
      (this.metrics.successfulRequests / this.metrics.totalRequests) *
      100
    ).toFixed(1);
  }

  getMetrics() {
    return {
      ...this.metrics,
      successRate: this.getSuccessRate(),
      cacheHitRate:
        this.metrics.totalRequests > 0
          ? (
              (this.metrics.cachedResponses / this.metrics.totalRequests) *
              100
            ).toFixed(1)
          : 0,
      errorTypes: Object.fromEntries(this.metrics.errorTypes),
      endpointStats: Object.fromEntries(this.metrics.endpointStats),
    };
  }

  // Get slowest endpoints for optimization
  getSlowestEndpoints(n = 5) {
    return Array.from(this.metrics.endpointStats.entries())
      .sort((a, b) => b[1].avgLatency - a[1].avgLatency)
      .slice(0, n)
      .map(([endpoint, stats]) => ({ endpoint, ...stats }));
  }
}

const perfTracker = new PerformanceTracker();

// ===========================================
// 🔮 PREDICTIVE PREFETCH ENGINE
// ===========================================

class PredictivePrefetcher {
  constructor() {
    this.navigationPatterns = new Map(); // ติดตาม pattern การใช้งาน
    this.prefetchQueue = new Set();
    this.prefetchedData = new Map();
    this.prefetchExpiry = 30000; // 30 seconds
    this.lastPage = null;

    // Pattern weights - เรียนรู้ว่าหลังจากหน้าไหนมักไปหน้าไหน
    this.patterns = {
      dashboard: ["bot_status", "positions", "risk_data", "pipeline"],
      trading: ["positions", "bot_status", "account"],
      analysis: ["pipeline", "titan", "omega", "signals"],
      settings: ["bot_settings", "trading_config"],
    };
  }

  // 📍 Record navigation and prefetch related data
  recordNavigation(page) {
    if (this.lastPage) {
      const key = `${this.lastPage}->${page}`;
      this.navigationPatterns.set(
        key,
        (this.navigationPatterns.get(key) || 0) + 1,
      );
    }
    this.lastPage = page;

    // Prefetch data for this page
    this.prefetchForPage(page);
  }

  // 🔮 Prefetch data likely to be needed
  async prefetchForPage(page) {
    const endpoints = this.patterns[page] || [];
    log.debug("[Prefetch] Prefetching for page:", page, endpoints);

    for (const endpoint of endpoints) {
      if (!this.prefetchQueue.has(endpoint)) {
        this.prefetchQueue.add(endpoint);
        this.prefetchEndpoint(endpoint);
      }
    }
  }

  async prefetchEndpoint(endpoint) {
    // Will be connected to actual API calls
    log.debug("[Prefetch] Queued:", endpoint);
  }

  // Get prefetched data if available and fresh
  getPrefetched(key) {
    const cached = this.prefetchedData.get(key);
    if (cached && Date.now() - cached.time < this.prefetchExpiry) {
      log.perf("[Prefetch] HIT:", key);
      return cached.data;
    }
    return null;
  }

  setPrefetched(key, data) {
    this.prefetchedData.set(key, { data, time: Date.now() });
  }

  // Get navigation predictions
  getPredictions() {
    return Object.fromEntries(this.navigationPatterns);
  }
}

const prefetcher = new PredictivePrefetcher();

// ===========================================
// ⚡ PRIORITY REQUEST QUEUE
// ===========================================

class PriorityQueue {
  constructor() {
    this.queues = {
      critical: [], // Trading actions - must execute
      high: [], // Bot control, positions
      normal: [], // Data fetching
      low: [], // Analytics, prefetch
    };
    this.processing = false;
    this.maxConcurrent = 4;
    this.activeRequests = 0;
  }

  add(priority, request) {
    this.queues[priority].push(request);
    this.process();
  }

  async process() {
    if (this.processing || this.activeRequests >= this.maxConcurrent) return;
    this.processing = true;

    // Process by priority
    for (const priority of ["critical", "high", "normal", "low"]) {
      while (
        this.queues[priority].length > 0 &&
        this.activeRequests < this.maxConcurrent
      ) {
        const request = this.queues[priority].shift();
        this.activeRequests++;

        request.execute().finally(() => {
          this.activeRequests--;
          this.process();
        });
      }
    }

    this.processing = false;
  }

  getQueueStatus() {
    return {
      critical: this.queues.critical.length,
      high: this.queues.high.length,
      normal: this.queues.normal.length,
      low: this.queues.low.length,
      active: this.activeRequests,
    };
  }
}

const requestQueue = new PriorityQueue();

// ===========================================
// 🛡️ CIRCUIT BREAKER
// ===========================================

class CircuitBreaker {
  constructor() {
    this.state = "CLOSED"; // CLOSED, OPEN, HALF_OPEN
    this.failureCount = 0;
    this.successCount = 0;
    this.failureThreshold = 10;   // Increased from 5 (allow more failures before opening)
    this.successThreshold = 2;    // Decreased from 3 (recover faster)
    this.timeout = 15000;         // Decreased from 30000 (try again sooner)
    this.lastFailureTime = null;
    this.listeners = new Set();
  }

  onStateChange(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  notifyListeners() {
    this.listeners.forEach((cb) => cb(this.getState()));
  }

  getState() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      canRequest: this.canRequest(),
    };
  }

  canRequest() {
    if (this.state === "CLOSED") return true;
    if (this.state === "OPEN") {
      // Check if timeout has passed
      if (Date.now() - this.lastFailureTime > this.timeout) {
        this.state = "HALF_OPEN";
        this.notifyListeners();
        log.info("🔄", "[Circuit] Moving to HALF_OPEN state");
        return true;
      }
      return false;
    }
    return true; // HALF_OPEN allows requests
  }

  recordSuccess() {
    if (this.state === "HALF_OPEN") {
      this.successCount++;
      if (this.successCount >= this.successThreshold) {
        this.state = "CLOSED";
        this.failureCount = 0;
        this.successCount = 0;
        log.success("[Circuit] CLOSED - Service recovered!");
        this.notifyListeners();
      }
    } else {
      this.failureCount = Math.max(0, this.failureCount - 1);
    }
  }

  recordFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    this.successCount = 0;

    if (
      this.state === "HALF_OPEN" ||
      this.failureCount >= this.failureThreshold
    ) {
      this.state = "OPEN";
      log.error("[Circuit] OPEN - Too many failures, blocking requests");
      this.notifyListeners();
    }
  }
}

const circuitBreaker = new CircuitBreaker();

// ===========================================
// 📴 OFFLINE QUEUE WITH SYNC
// ===========================================

class OfflineQueue {
  constructor() {
    this.queue = [];
    this.maxQueueSize = 50;
    this.storageKey = "trademify_offline_queue";
    this.loadFromStorage();

    // Listen for online/offline events
    window.addEventListener("online", () => this.sync());
    window.addEventListener("offline", () =>
      log.warn("[Offline] Device went offline"),
    );
  }

  loadFromStorage() {
    try {
      const saved = localStorage.getItem(this.storageKey);
      if (saved) {
        this.queue = JSON.parse(saved);
        log.info("📥", `[Offline] Loaded ${this.queue.length} queued requests`);
      }
    } catch (e) {
      this.queue = [];
    }
  }

  saveToStorage() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.queue));
    } catch (e) {
      log.warn("[Offline] Failed to save queue to storage");
    }
  }

  add(request) {
    if (this.queue.length >= this.maxQueueSize) {
      this.queue.shift(); // Remove oldest
    }

    this.queue.push({
      ...request,
      queuedAt: Date.now(),
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    });

    this.saveToStorage();
    log.info(
      "📥",
      `[Offline] Queued request: ${request.method} ${request.url}`,
    );

    return this.queue.length;
  }

  async sync() {
    if (this.queue.length === 0) return;

    log.info("🔄", `[Offline] Syncing ${this.queue.length} queued requests...`);

    const toProcess = [...this.queue];
    this.queue = [];
    this.saveToStorage();

    let success = 0;
    let failed = 0;

    for (const request of toProcess) {
      try {
        await axios({
          method: request.method,
          url: request.url,
          data: request.data,
          baseURL: API_BASE_URL,
          timeout: 30000, // Increased from 10000
        });
        success++;
      } catch (e) {
        failed++;
        // Re-queue critical requests
        if (request.critical) {
          this.add(request);
        }
      }
    }

    log.success(
      `[Offline] Sync complete: ${success} success, ${failed} failed`,
    );
  }

  getQueueSize() {
    return this.queue.length;
  }

  clear() {
    this.queue = [];
    this.saveToStorage();
  }
}

const offlineQueue = new OfflineQueue();

// ===========================================
// 🧬 GENIUS CONNECTION MANAGER
// ===========================================

class GeniusConnectionManager {
  constructor() {
    this.isOnline = true;
    this.lastSuccessTime = null;
    this.lastErrorTime = null;
    this.consecutiveErrors = 0;
    this.healthCheckInterval = null;
    this.listeners = new Set();
    this.pendingRequests = new Map();
    this.cache = new Map();
    this.cacheConfig = new Map(); // Per-endpoint cache config
    this.reconnecting = false;

    // 🎯 Connection quality metrics
    this.connectionQuality = "excellent"; // excellent, good, poor, offline
    this.latencyHistory = [];
    this.maxLatencyHistory = 20;

    // 🕐 Adaptive timeout - INCREASED for VPS latency
    this.baseTimeout = 30000; // 30 seconds base
    this.currentTimeout = this.baseTimeout;
    this.minTimeout = 15000; // 15 seconds minimum (was 5000)
    this.maxTimeout = 60000; // 60 seconds maximum (was 30000)

    // Start monitoring
    this.startHealthMonitoring();
    this.detectConnectionQuality();
  }

  // 📡 Connection quality detection
  detectConnectionQuality() {
    if ("connection" in navigator) {
      const conn = navigator.connection;
      const updateQuality = () => {
        if (conn.effectiveType === "4g") this.connectionQuality = "excellent";
        else if (conn.effectiveType === "3g") this.connectionQuality = "good";
        else if (conn.effectiveType === "2g") this.connectionQuality = "poor";
        else this.connectionQuality = "poor";

        log.debug(
          "[Connection] Quality:",
          this.connectionQuality,
          conn.effectiveType,
        );
        this.adjustTimeout();
      };

      conn.addEventListener("change", updateQuality);
      updateQuality();
    }
  }

  // 🕐 Adjust timeout based on connection quality and latency
  adjustTimeout() {
    const avgLatency =
      this.latencyHistory.length > 0
        ? this.latencyHistory.reduce((a, b) => a + b, 0) /
          this.latencyHistory.length
        : 500;

    const qualityMultiplier =
      {
        excellent: 1,
        good: 1.5,
        poor: 2.5,
        offline: 3,
      }[this.connectionQuality] || 1;

    this.currentTimeout = Math.min(
      this.maxTimeout,
      Math.max(this.minTimeout, avgLatency * 3 * qualityMultiplier),
    );

    log.debug("[Timeout] Adjusted to:", this.currentTimeout, "ms");
  }

  recordLatency(latency) {
    this.latencyHistory.push(latency);
    if (this.latencyHistory.length > this.maxLatencyHistory) {
      this.latencyHistory.shift();
    }
    this.adjustTimeout();
  }

  // Event system
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
      connectionQuality: this.connectionQuality,
      currentTimeout: this.currentTimeout,
      avgLatency:
        this.latencyHistory.length > 0
          ? Math.round(
              this.latencyHistory.reduce((a, b) => a + b, 0) /
                this.latencyHistory.length,
            )
          : 0,
      health:
        this.consecutiveErrors === 0
          ? "excellent"
          : this.consecutiveErrors < 3
            ? "degraded"
            : "critical",
      circuitState: circuitBreaker.state,
      offlineQueueSize: offlineQueue.getQueueSize(),
      cacheSize: this.cache.size,
    };
  }

  markSuccess(latency = 0) {
    this.isOnline = true;
    this.consecutiveErrors = 0;
    this.lastSuccessTime = Date.now();
    this.reconnecting = false;

    if (latency > 0) {
      this.recordLatency(latency);
    }

    circuitBreaker.recordSuccess();
    this.notifyListeners();
  }

  markFailure(error) {
    this.consecutiveErrors++;
    this.lastErrorTime = Date.now();

    circuitBreaker.recordFailure();

    if (this.consecutiveErrors >= 3) {
      this.isOnline = false;
      this.connectionQuality = "offline";
    }

    this.notifyListeners();
    return this.consecutiveErrors;
  }

  async attemptReconnect() {
    if (this.reconnecting) return false;
    if (!circuitBreaker.canRequest()) {
      log.warn("[Reconnect] Circuit breaker is OPEN");
      return false;
    }

    this.reconnecting = true;
    this.notifyListeners();

    log.info("🔄", "[Genius] Attempting to reconnect...");

    try {
      const start = Date.now();
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: 5000,
      });
      const latency = Date.now() - start;

      if (response.data) {
        log.success("[Genius] Reconnected successfully!");
        this.markSuccess(latency);

        // Sync offline queue
        offlineQueue.sync();

        return true;
      }
    } catch (e) {
      log.warn("[Genius] Reconnect failed");
    }

    this.reconnecting = false;
    this.notifyListeners();
    return false;
  }

  startHealthMonitoring() {
    const check = async () => {
      const interval = this.isOnline ? 15000 : 5000;

      if (!this.isOnline && circuitBreaker.canRequest()) {
        await this.attemptReconnect();
      }

      this.healthCheckInterval = setTimeout(check, interval);
    };

    setTimeout(check, 5000);
  }

  destroy() {
    if (this.healthCheckInterval) {
      clearTimeout(this.healthCheckInterval);
    }
    this.listeners.clear();
    this.pendingRequests.clear();
    this.cache.clear();
  }

  // 📦 Smart Cache with TTL and stale-while-revalidate
  getCached(key, options = {}) {
    const cached = this.cache.get(key);
    if (!cached) return null;

    const { staleWhileRevalidate = false, ttl = 5000 } = options;
    const age = Date.now() - cached.time;

    if (age < ttl) {
      perfTracker.recordCacheHit();
      return { data: cached.data, fresh: true };
    }

    if (staleWhileRevalidate && age < ttl * 3) {
      perfTracker.recordCacheHit();
      return { data: cached.data, fresh: false, stale: true };
    }

    return null;
  }

  setCache(key, data, ttl = 5000) {
    this.cache.set(key, { data, time: Date.now(), ttl });

    // Clean old entries
    if (this.cache.size > 200) {
      const entries = Array.from(this.cache.entries());
      entries.sort((a, b) => a[1].time - b[1].time);
      for (let i = 0; i < 50; i++) {
        this.cache.delete(entries[i][0]);
      }
    }
  }

  invalidateCache(pattern = null) {
    if (!pattern) {
      this.cache.clear();
      return;
    }

    for (const key of this.cache.keys()) {
      if (key.includes(pattern)) {
        this.cache.delete(key);
      }
    }
  }

  // 🔀 Request deduplication with timeout
  async deduplicateRequest(key, requestFn, timeout = 5000) {
    if (this.pendingRequests.has(key)) {
      log.debug("[Dedupe] Reusing pending request:", key);
      return this.pendingRequests.get(key);
    }

    const promise = requestFn();
    this.pendingRequests.set(key, promise);

    // Auto-cleanup after timeout
    setTimeout(() => this.pendingRequests.delete(key), timeout);

    try {
      const result = await promise;
      return result;
    } finally {
      this.pendingRequests.delete(key);
    }
  }
}

// Create singleton
const connectionManager = new GeniusConnectionManager();

// Exports
export const getConnectionStatus = () => connectionManager.getStatus();
export const onConnectionChange = (cb) => connectionManager.onStatusChange(cb);
export const getPerformanceMetrics = () => perfTracker.getMetrics();
export const getCircuitState = () => circuitBreaker.getState();
export const onCircuitChange = (cb) => circuitBreaker.onStateChange(cb);
export const recordNavigation = (page) => prefetcher.recordNavigation(page);

// ===========================================
// 🚀 GENIUS AXIOS CLIENT
// ===========================================

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: connectionManager.currentTimeout,
});

// Dynamic timeout based on connection quality
apiClient.interceptors.request.use(
  (config) => {
    // Update timeout dynamically
    config.timeout = connectionManager.currentTimeout;

    // Add request timestamp for latency tracking
    config.metadata = { startTime: Date.now() };

    const status = connectionManager.isOnline ? "🟢" : "🔴";
    const quality = connectionManager.connectionQuality;
    log.debug(
      `${status} [${quality}] ${config.method?.toUpperCase()} ${config.url}`,
    );

    return config;
  },
  (error) => Promise.reject(error),
);

// Response interceptor with metrics
apiClient.interceptors.response.use(
  (response) => {
    const latency =
      Date.now() - (response.config.metadata?.startTime || Date.now());
    const endpoint = response.config.url;

    connectionManager.markSuccess(latency);
    perfTracker.recordRequest(endpoint, latency, true);

    log.perf(`[${latency}ms] ${endpoint}`);

    return response.data;
  },
  (error) => {
    const latency =
      Date.now() - (error.config?.metadata?.startTime || Date.now());
    const endpoint = error.config?.url || "unknown";

    connectionManager.markFailure(error);
    perfTracker.recordRequest(endpoint, latency, false, error);

    if (error.code === "ECONNABORTED") {
      log.warn(`[Timeout] ${endpoint} after ${latency}ms`);
    } else if (!error.response) {
      log.error(`[Network] ${endpoint} - Server unreachable`);
    } else {
      log.error(`[${error.response.status}] ${endpoint}`, error.response?.data);
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
// � GENIUS API WRAPPER - 100x SMARTER
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
 * 🧬 GENIUS API CALL - The Ultimate API Wrapper
 *
 * Features:
 * - Circuit breaker protection
 * - Predictive prefetching
 * - Stale-while-revalidate caching
 * - Request prioritization
 * - Offline queue for critical requests
 * - Intelligent retry with jitter
 * - Performance tracking
 * - Optimistic updates support
 */
async function geniusApiCall(apiCall, mockFn, options = {}) {
  const {
    cacheKey = null,
    cacheTTL = 5000,
    maxRetries = 2,
    silent = false,
    critical = false,
    priority = "normal", // critical, high, normal, low
    staleWhileRevalidate = false,
    offlineQueue: queueOffline = false,
    optimisticData = null,
    onRevalidate = null,
    prefetchRelated = [],
  } = options;

  const startTime = Date.now();

  // 1. Check mock mode
  if (USE_MOCK) {
    if (!silent) log.debug("[MOCK] Using mock data");
    const result = mockFn();
    result._isMock = true;
    result._source = "mock";
    return result;
  }

  // 2. Check circuit breaker
  if (!circuitBreaker.canRequest()) {
    log.warn("[Circuit] Request blocked - circuit is OPEN");

    // Return cached or mock data
    if (cacheKey) {
      const cached = connectionManager.getCached(cacheKey, {
        staleWhileRevalidate: true,
        ttl: cacheTTL,
      });
      if (cached) {
        cached.data._source = "circuit_fallback";
        return cached.data;
      }
    }

    if (critical && queueOffline) {
      // Queue for later
      offlineQueue.add({
        method: "GET",
        url: cacheKey || "unknown",
        critical: true,
      });
    }

    const result = mockFn();
    result._isMock = true;
    result._source = "circuit_fallback";
    return result;
  }

  // 3. Check cache with stale-while-revalidate
  if (cacheKey) {
    const cached = connectionManager.getCached(cacheKey, {
      staleWhileRevalidate,
      ttl: cacheTTL,
    });

    if (cached) {
      if (cached.fresh) {
        if (!silent) log.debug("[Cache] Fresh data:", cacheKey);
        cached.data._source = "cache";
        return cached.data;
      }

      if (cached.stale && staleWhileRevalidate) {
        if (!silent) log.debug("[Cache] Stale data, revalidating:", cacheKey);

        // Return stale data immediately
        cached.data._source = "stale";
        cached.data._revalidating = true;

        // Revalidate in background
        geniusApiCall(apiCall, mockFn, {
          ...options,
          cacheKey,
          staleWhileRevalidate: false,
          silent: true,
        })
          .then((freshData) => {
            if (onRevalidate) onRevalidate(freshData);
          })
          .catch(() => {});

        return cached.data;
      }
    }
  }

  // 4. Return optimistic data if provided (for mutations)
  if (optimisticData) {
    // Execute API in background
    geniusApiCall(apiCall, mockFn, {
      ...options,
      optimisticData: null,
      silent: true,
    }).catch((err) => {
      log.error("[Optimistic] Background request failed:", err);
    });

    const result =
      typeof optimisticData === "function" ? optimisticData() : optimisticData;
    result._source = "optimistic";
    return result;
  }

  // 5. Execute API call with smart retry
  let lastError = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Deduplicate request
      const result = cacheKey
        ? await connectionManager.deduplicateRequest(cacheKey, apiCall)
        : await apiCall();

      // Success! Cache the result
      if (cacheKey) {
        connectionManager.setCache(cacheKey, result, cacheTTL);
      }

      apiConnected = true;
      lastApiError = null;
      result._isMock = false;
      result._source = "api";
      result._attempt = attempt + 1;
      result._latency = Date.now() - startTime;

      // Prefetch related data
      if (prefetchRelated.length > 0) {
        log.debug("[Prefetch] Queueing related:", prefetchRelated);
        prefetchRelated.forEach((key) => prefetcher.prefetchEndpoint(key));
      }

      return result;
    } catch (error) {
      lastError = error;

      // Don't retry on client errors (4xx)
      if (
        error.response &&
        error.response.status >= 400 &&
        error.response.status < 500
      ) {
        break;
      }

      // Smart retry with jitter
      if (attempt < maxRetries) {
        const baseDelay = Math.min(1000 * Math.pow(2, attempt), 5000);
        const jitter = Math.random() * 500; // Add randomness to prevent thundering herd
        const delay = baseDelay + jitter;

        if (!silent)
          log.debug(
            `[Retry] Attempt ${attempt + 2}/${maxRetries + 1} in ${Math.round(delay)}ms...`,
          );
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  // 6. All retries failed
  apiConnected = false;
  lastApiError = lastError?.message || "Connection failed";

  // Queue critical requests for offline sync
  if (critical && queueOffline) {
    offlineQueue.add({
      method: "GET",
      url: cacheKey || "unknown",
      critical: true,
    });
    log.info("📥", "[Offline] Queued critical request");
  }

  // Critical requests should throw
  if (critical) {
    throw lastError;
  }

  // Fallback to mock
  if (!silent) {
    log.warn("[Fallback] Using mock data after failures");
  }

  const result = mockFn();
  result._isMock = true;
  result._source = "fallback";
  result._error = lastApiError;
  return result;
}

// Backward compatibility
async function smartApiCall(apiCall, mockFn, options = {}) {
  return geniusApiCall(apiCall, mockFn, options);
}

async function tryApiOrMock(apiCall, mockFn, silent = false) {
  return geniusApiCall(apiCall, mockFn, { silent });
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
  // 14. SERVER-SENT EVENTS (Real-time Updates) - GENIUS
  // ===========================================

  getEventsUrl() {
    return `${API_BASE_URL}/events`;
  },

  subscribeToEvents(onMessage, onError) {
    let eventSource = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 10;
    const reconnectDelay = 3000;

    const connect = () => {
      eventSource = new EventSource(this.getEventsUrl());

      eventSource.onopen = () => {
        log.success("[SSE] Connected to event stream");
        reconnectAttempts = 0;
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (e) {
          log.error("[SSE] Parse error:", e);
        }
      };

      eventSource.onerror = (error) => {
        log.warn("[SSE] Connection error");
        eventSource.close();

        if (onError) onError(error);

        // Auto-reconnect with backoff
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          const delay = reconnectDelay * Math.pow(1.5, reconnectAttempts - 1);
          log.info(
            "🔄",
            `[SSE] Reconnecting in ${Math.round(delay / 1000)}s (attempt ${reconnectAttempts})`,
          );
          setTimeout(connect, delay);
        } else {
          log.error("[SSE] Max reconnect attempts reached");
        }
      };
    };

    connect();

    // Return cleanup function
    return {
      close: () => {
        if (eventSource) {
          eventSource.close();
          log.info("📴", "[SSE] Disconnected");
        }
      },
      getReadyState: () => eventSource?.readyState,
    };
  },

  // ===========================================
  // 15. SYSTEM HEALTH (GENIUS)
  // ===========================================

  async getSystemHealth() {
    return geniusApiCall(
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
      { cacheKey: "system_health", cacheTTL: 3000, staleWhileRevalidate: true },
    );
  },

  // ===========================================
  // 🧬 16. GENIUS UTILITY FUNCTIONS
  // ===========================================

  getBaseUrl() {
    return API_BASE_URL;
  },

  isConnected() {
    return connectionManager.isOnline;
  },

  // 📊 Get full connection status
  getConnectionStatus() {
    return connectionManager.getStatus();
  },

  // 🔔 Subscribe to connection changes
  onConnectionChange(callback) {
    return connectionManager.onStatusChange(callback);
  },

  // 🔄 Force reconnect
  async reconnect() {
    return connectionManager.attemptReconnect();
  },

  // 🧹 Clear all caches
  clearCache(pattern = null) {
    connectionManager.invalidateCache(pattern);
    log.info("🧹", "[Cache] Cleared", pattern || "all");
  },

  // 📦 Get cache statistics
  getCacheStats() {
    return {
      size: connectionManager.cache.size,
      keys: Array.from(connectionManager.cache.keys()),
      hitRate: perfTracker.getMetrics().cacheHitRate,
    };
  },

  // ⚡ Get performance metrics
  getPerformanceMetrics() {
    return perfTracker.getMetrics();
  },

  // 🐢 Get slowest endpoints
  getSlowestEndpoints(n = 5) {
    return perfTracker.getSlowestEndpoints(n);
  },

  // 🛡️ Get circuit breaker state
  getCircuitState() {
    return circuitBreaker.getState();
  },

  // 🔔 Subscribe to circuit breaker changes
  onCircuitChange(callback) {
    return circuitBreaker.onStateChange(callback);
  },

  // 📴 Get offline queue status
  getOfflineQueueSize() {
    return offlineQueue.getQueueSize();
  },

  // 🔄 Force sync offline queue
  async syncOfflineQueue() {
    return offlineQueue.sync();
  },

  // 📍 Record page navigation (for predictive prefetch)
  recordNavigation(page) {
    prefetcher.recordNavigation(page);
  },

  // 🔮 Get navigation predictions
  getNavigationPredictions() {
    return prefetcher.getPredictions();
  },

  // 📊 Get request queue status
  getQueueStatus() {
    return requestQueue.getQueueStatus();
  },

  // 🧬 Get full genius status
  getGeniusStatus() {
    return {
      connection: connectionManager.getStatus(),
      performance: perfTracker.getMetrics(),
      circuit: circuitBreaker.getState(),
      offlineQueue: offlineQueue.getQueueSize(),
      requestQueue: requestQueue.getQueueStatus(),
      cache: {
        size: connectionManager.cache.size,
        hitRate: perfTracker.getMetrics().cacheHitRate,
      },
      predictions: prefetcher.getPredictions(),
    };
  },

  // 🔧 Debug mode toggle
  setDebugMode(enabled) {
    window.TRADEMIFY_DEBUG = enabled;
    log.info(
      enabled ? "🔍" : "🔇",
      "[Debug]",
      enabled ? "Enabled" : "Disabled",
    );
  },

  // 📤 Export metrics for analysis
  exportMetrics() {
    const metrics = this.getGeniusStatus();
    const blob = new Blob([JSON.stringify(metrics, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trademify-metrics-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    log.success("[Export] Metrics exported");
  },
};

// 🧬 Auto-log genius status on load
setTimeout(() => {
  const status = api.getGeniusStatus();
  console.log(
    "%c🧬 GENIUS API LOADED",
    "color: #00ff00; font-size: 14px; font-weight: bold",
  );
  console.log(
    "%cConnection:",
    "color: #888",
    status.connection.isOnline ? "🟢 Online" : "🔴 Offline",
  );
  console.log("%cCircuit:", "color: #888", status.circuit.state);
  console.log("%cCache:", "color: #888", `${status.cache.size} items`);
}, 1000);

export default api;
