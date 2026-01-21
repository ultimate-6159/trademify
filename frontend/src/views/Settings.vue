<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-3xl font-bold text-white">Settings</h2>
        <p class="text-gray-400 mt-1">Configure trading and pattern matching parameters</p>
      </div>
      <div class="flex items-center space-x-2">
        <span 
          :class="['w-3 h-3 rounded-full', isConnected ? 'bg-success' : 'bg-danger']"
        ></span>
        <span class="text-gray-400">{{ isConnected ? 'Connected' : 'Disconnected' }}</span>
      </div>
    </div>

    <!-- Save Notification -->
    <div 
      v-if="saveMessage"
      :class="[
        'p-4 rounded-lg flex items-center justify-between',
        saveSuccess ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'
      ]"
    >
      <span>{{ saveMessage }}</span>
      <button @click="saveMessage = ''" class="text-xl">&times;</button>
    </div>
    
    <!-- Pattern Matching Settings -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
        <svg class="w-5 h-5 mr-2 text-primary" fill="currentColor" viewBox="0 0 20 20">
          <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
        </svg>
        Pattern Matching Configuration
      </h3>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label class="block text-gray-400 mb-2">Window Size</label>
          <input 
            type="number" 
            v-model.number="settings.windowSize"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
          <p class="text-gray-500 text-sm mt-1">Number of candles per pattern (default: 60)</p>
        </div>
        
        <div>
          <label class="block text-gray-400 mb-2">Confidence Threshold (%)</label>
          <input 
            type="number" 
            v-model.number="settings.confidenceThreshold"
            min="50"
            max="100"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
          <p class="text-gray-500 text-sm mt-1">Minimum confidence for signal (70%)</p>
        </div>
        
        <div>
          <label class="block text-gray-400 mb-2">Minimum Correlation</label>
          <input 
            type="number" 
            v-model.number="settings.minCorrelation"
            min="0"
            max="1"
            step="0.01"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
          <p class="text-gray-500 text-sm mt-1">Pattern similarity threshold (0.85)</p>
        </div>
        
        <div>
          <label class="block text-gray-400 mb-2">Top K Patterns</label>
          <input 
            type="number" 
            v-model.number="settings.topK"
            min="5"
            max="100"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
          <p class="text-gray-500 text-sm mt-1">Similar patterns to analyze (5-100)</p>
        </div>
      </div>
    </div>
    
    <!-- Trading Risk Settings -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
        <svg class="w-5 h-5 mr-2 text-warning" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
        </svg>
        Risk Management
      </h3>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label class="block text-gray-400 mb-2">Max Risk per Trade (%)</label>
          <input 
            type="number" 
            v-model.number="settings.maxRiskPerTrade"
            min="0.5"
            max="10"
            step="0.5"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
          <p class="text-gray-500 text-sm mt-1">% of balance per trade (2%)</p>
        </div>
        
        <div>
          <label class="block text-gray-400 mb-2">Max Daily Loss (%)</label>
          <input 
            type="number" 
            v-model.number="settings.maxDailyLoss"
            min="1"
            max="20"
            step="0.5"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
          <p class="text-gray-500 text-sm mt-1">Stop trading if loss exceeds (5%)</p>
        </div>
        
        <div>
          <label class="block text-gray-400 mb-2">Max Concurrent Positions</label>
          <input 
            type="number" 
            v-model.number="settings.maxPositions"
            min="1"
            max="20"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
          <p class="text-gray-500 text-sm mt-1">Max open positions (5)</p>
        </div>
        
        <div>
          <label class="block text-gray-400 mb-2">Max Drawdown (%)</label>
          <input 
            type="number" 
            v-model.number="settings.maxDrawdown"
            min="5"
            max="30"
            step="1"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
          <p class="text-gray-500 text-sm mt-1">Stop all trading if exceeded (10%)</p>
        </div>
      </div>
    </div>

    <!-- Signal Settings -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
        <svg class="w-5 h-5 mr-2 text-success" fill="currentColor" viewBox="0 0 20 20">
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
        </svg>
        Signal Settings
      </h3>
      
      <div class="space-y-4">
        <div>
          <label class="block text-gray-400 mb-2">Allowed Signals</label>
          <div class="flex flex-wrap gap-3">
            <label 
              v-for="signal in availableSignals" 
              :key="signal"
              class="flex items-center space-x-2 cursor-pointer"
            >
              <input 
                type="checkbox"
                :value="signal"
                v-model="settings.allowedSignals"
                class="w-4 h-4 rounded bg-dark-200 border-gray-700 text-primary focus:ring-primary"
              />
              <span :class="getSignalClass(signal)">{{ signal }}</span>
            </label>
          </div>
          <p class="text-gray-500 text-sm mt-2">Which signals should trigger trades</p>
        </div>
        
        <div class="flex items-center justify-between pt-4 border-t border-gray-700">
          <div>
            <label class="text-gray-400">Paper Trading Mode</label>
            <p class="text-gray-500 text-sm">Test without real money</p>
          </div>
          <button 
            @click="settings.paperTrading = !settings.paperTrading"
            :class="[
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
              settings.paperTrading ? 'bg-primary' : 'bg-gray-600'
            ]"
          >
            <span 
              :class="[
                'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                settings.paperTrading ? 'translate-x-6' : 'translate-x-1'
              ]"
            />
          </button>
        </div>
      </div>
    </div>

    <!-- Trading Symbols -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
        <svg class="w-5 h-5 mr-2 text-primary" fill="currentColor" viewBox="0 0 20 20">
          <path d="M4 4a2 2 0 00-2 2v4a2 2 0 002 2V6h10a2 2 0 00-2-2H4zm2 6a2 2 0 012-2h8a2 2 0 012 2v4a2 2 0 01-2 2H8a2 2 0 01-2-2v-4zm6 4a2 2 0 100-4 2 2 0 000 4z"/>
        </svg>
        Trading Symbols
      </h3>
      
      <div class="space-y-4">
        <div class="flex flex-wrap gap-2">
          <span 
            v-for="symbol in settings.symbols" 
            :key="symbol"
            class="px-3 py-1 bg-primary/20 text-primary rounded-full flex items-center"
          >
            {{ symbol }}
            <button 
              @click="removeSymbol(symbol)"
              class="ml-2 text-primary/60 hover:text-primary"
            >
              &times;
            </button>
          </span>
        </div>
        
        <div class="flex items-center space-x-2">
          <input 
            type="text"
            v-model="newSymbol"
            @keyup.enter="addSymbol"
            placeholder="Add symbol (e.g., EURUSD)"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white flex-1 focus:outline-none focus:border-primary"
          />
          <button @click="addSymbol" class="btn btn-primary">Add</button>
        </div>
      </div>
    </div>

    <!-- API Connection -->
    <div class="card">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
        <svg class="w-5 h-5 mr-2 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M2 5a2 2 0 012-2h12a2 2 0 012 2v10a2 2 0 01-2 2H4a2 2 0 01-2-2V5zm3.293 1.293a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 01-1.414-1.414L7.586 10 5.293 7.707a1 1 0 010-1.414zM11 12a1 1 0 100 2h3a1 1 0 100-2h-3z" clip-rule="evenodd"/>
        </svg>
        API Configuration
      </h3>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label class="block text-gray-400 mb-2">Backend URL</label>
          <input 
            type="text"
            v-model="settings.apiUrl"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          />
        </div>
        
        <div>
          <label class="block text-gray-400 mb-2">Timeframe</label>
          <select 
            v-model="settings.timeframe"
            class="bg-dark-200 border border-gray-700 rounded-lg px-4 py-2 text-white w-full focus:outline-none focus:border-primary"
          >
            <option value="M1">M1</option>
            <option value="M5">M5</option>
            <option value="M15">M15</option>
            <option value="H1">H1</option>
            <option value="H4">H4</option>
            <option value="D1">Daily</option>
          </select>
        </div>
      </div>
      
      <div class="mt-4 flex items-center space-x-4">
        <button @click="testConnection" :disabled="isTesting" class="btn btn-secondary">
          {{ isTesting ? 'Testing...' : 'Test Connection' }}
        </button>
        <span 
          v-if="connectionStatus" 
          :class="connectionStatus === 'success' ? 'text-success' : 'text-danger'"
        >
          {{ connectionStatus === 'success' ? '✓ Connected' : '✗ Connection failed' }}
        </span>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="flex items-center justify-between">
      <button @click="resetToDefaults" class="btn btn-secondary">
        Reset to Defaults
      </button>
      <div class="flex space-x-3">
        <button @click="loadSettings" class="btn btn-secondary">
          Reload
        </button>
        <button @click="saveSettings" :disabled="isSaving" class="btn btn-primary">
          {{ isSaving ? 'Saving...' : 'Save Settings' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useTradingStore } from '../stores/trading'

const tradingStore = useTradingStore()

// Auto-detect API URL based on current host
const getDefaultApiUrl = () => {
  // If explicitly set in environment
  const envUrl = import.meta.env.VITE_API_URL
  if (envUrl) {
    // Remove /api/v1 suffix if present
    return envUrl.replace(/\/api\/v1\/?$/, '')
  }
  
  // If running on localhost (dev machine)
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000'
  }
  
  // If running on VPS or remote server, use same host with http
  return `http://${window.location.hostname}:8000`
}

const API_BASE = getDefaultApiUrl()

const isConnected = ref(false)
const isSaving = ref(false)
const isTesting = ref(false)
const saveMessage = ref('')
const saveSuccess = ref(false)
const connectionStatus = ref(null)
const newSymbol = ref('')

const availableSignals = ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']

const settings = ref({
  // Pattern matching
  windowSize: 60,
  confidenceThreshold: 70,
  minCorrelation: 0.85,
  topK: 10,
  
  // Risk management
  maxRiskPerTrade: 2.0,
  maxDailyLoss: 5.0,
  maxPositions: 5,
  maxDrawdown: 10.0,
  
  // Signal settings
  allowedSignals: ['STRONG_BUY', 'STRONG_SELL'],
  paperTrading: true,
  
  // Symbols
  symbols: ['EURUSD', 'GBPUSD', 'XAUUSD'],
  
  // API
  apiUrl: API_BASE,
  timeframe: 'H1'
})

function getSignalClass(signal) {
  if (signal.includes('STRONG_BUY')) return 'text-success font-semibold'
  if (signal.includes('BUY')) return 'text-green-400'
  if (signal.includes('STRONG_SELL')) return 'text-danger font-semibold'
  if (signal.includes('SELL')) return 'text-red-400'
  return 'text-gray-400'
}

function addSymbol() {
  const symbol = newSymbol.value.trim().toUpperCase()
  if (symbol && !settings.value.symbols.includes(symbol)) {
    settings.value.symbols.push(symbol)
    newSymbol.value = ''
  }
}

function removeSymbol(symbol) {
  settings.value.symbols = settings.value.symbols.filter(s => s !== symbol)
}

async function loadSettings() {
  try {
    const baseUrl = settings.value.apiUrl.replace(/\/api\/v1\/?$/, '')
    const response = await fetch(`${baseUrl}/api/v1/trading/settings`)
    if (response.ok) {
      const data = await response.json()
      
      // Update settings from server (ถ้ามีค่าจาก server ก็ใช้ ไม่มีก็ใช้ค่าเดิมจาก localStorage)
      // Paper trading
      if (data.paper_trading !== undefined) settings.value.paperTrading = data.paper_trading
      
      // Risk settings
      if (data.risk) {
        if (data.risk.max_risk_per_trade !== undefined) settings.value.maxRiskPerTrade = data.risk.max_risk_per_trade
        if (data.risk.max_daily_loss !== undefined) settings.value.maxDailyLoss = data.risk.max_daily_loss
        if (data.risk.max_positions !== undefined) settings.value.maxPositions = data.risk.max_positions
        if (data.risk.max_drawdown !== undefined) settings.value.maxDrawdown = data.risk.max_drawdown
      }
      
      // Signal settings
      if (data.signals) {
        if (data.signals.min_confidence !== undefined) settings.value.confidenceThreshold = data.signals.min_confidence
        if (data.signals.allowed_signals !== undefined) settings.value.allowedSignals = data.signals.allowed_signals
      }
      
      // Pattern settings
      if (data.pattern) {
        if (data.pattern.top_k !== undefined) settings.value.topK = data.pattern.top_k
        if (data.pattern.min_correlation !== undefined) settings.value.minCorrelation = data.pattern.min_correlation
        if (data.pattern.window_size !== undefined) settings.value.windowSize = data.pattern.window_size
      }
      
      // Symbols and timeframe
      if (data.trading_symbols !== undefined) settings.value.symbols = data.trading_symbols
      if (data.timeframe !== undefined) settings.value.timeframe = data.timeframe
      
      isConnected.value = true
      console.log('[Settings] Loaded from server:', settings.value)
    }
  } catch (error) {
    console.error('Failed to load settings:', error)
    isConnected.value = false
  }
}

async function saveSettings() {
  isSaving.value = true
  saveMessage.value = ''
  
  try {
    const payload = {
      paper_trading: settings.value.paperTrading,
      risk: {
        max_risk_per_trade: settings.value.maxRiskPerTrade,
        max_daily_loss: settings.value.maxDailyLoss,
        max_positions: settings.value.maxPositions
      },
      signals: {
        min_confidence: settings.value.confidenceThreshold,
        allowed_signals: settings.value.allowedSignals
      },
      pattern: {
        top_k: settings.value.topK,
        min_correlation: settings.value.minCorrelation,
        window_size: settings.value.windowSize
      },
      symbols: settings.value.symbols,
      timeframe: settings.value.timeframe
    }
    
    // Always save to localStorage as fallback
    localStorage.setItem('trademify_settings', JSON.stringify(settings.value))
    
    const baseUrl = settings.value.apiUrl.replace(/\/api\/v1\/?$/, '')
    const response = await fetch(`${baseUrl}/api/v1/trading/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    
    if (response.ok) {
      saveMessage.value = '✓ Settings saved successfully!'
      saveSuccess.value = true
      
      // Reload trading store settings
      await tradingStore.fetchSettings()
    } else {
      throw new Error('Failed to save to server')
    }
  } catch (error) {
    console.error('Failed to save settings to server:', error)
    // Still show success because we saved to localStorage
    saveMessage.value = '✓ Settings saved locally (server unavailable)'
    saveSuccess.value = true
  } finally {
    isSaving.value = false
    
    // Clear message after 5 seconds
    setTimeout(() => {
      saveMessage.value = ''
    }, 5000)
  }
}

async function testConnection() {
  isTesting.value = true
  connectionStatus.value = null
  
  try {
    // Remove /api/v1 suffix if present, then add /api/v1/health
    const baseUrl = settings.value.apiUrl.replace(/\/api\/v1\/?$/, '')
    const response = await fetch(`${baseUrl}/api/v1/health`)
    if (response.ok) {
      connectionStatus.value = 'success'
      isConnected.value = true
    } else {
      connectionStatus.value = 'failed'
      isConnected.value = false
    }
  } catch (error) {
    console.error('Connection test failed:', error)
    connectionStatus.value = 'failed'
    isConnected.value = false
  } finally {
    isTesting.value = false
  }
}

function resetToDefaults() {
  settings.value = {
    windowSize: 60,
    confidenceThreshold: 70,
    minCorrelation: 0.85,
    topK: 10,
    maxRiskPerTrade: 2.0,
    maxDailyLoss: 5.0,
    maxPositions: 5,
    maxDrawdown: 10.0,
    allowedSignals: ['STRONG_BUY', 'STRONG_SELL'],
    paperTrading: true,
    symbols: ['EURUSD', 'GBPUSD', 'XAUUSD'],
    apiUrl: API_BASE,
    timeframe: 'H1'
  }
  localStorage.removeItem('trademify_settings')
  saveMessage.value = 'Settings reset to defaults (not saved yet)'
  saveSuccess.value = true
}

// Load settings from localStorage on mount
function loadFromLocalStorage() {
  const saved = localStorage.getItem('trademify_settings')
  if (saved) {
    try {
      const parsed = JSON.parse(saved)
      // Merge with defaults to handle new fields
      settings.value = { ...settings.value, ...parsed }
      console.log('[Settings] Loaded from localStorage:', settings.value)
    } catch (e) {
      console.error('[Settings] Failed to parse localStorage:', e)
    }
  }
}

onMounted(() => {
  loadFromLocalStorage()  // First load from localStorage (instant)
  loadSettings()          // Then try to load from server
  testConnection()
  
  // เริ่ม Auto-Sync - ดึงข้อมูลจาก Backend ทุก 10 วินาที
  // เพื่อให้ทุกอุปกรณ์เห็นข้อมูลตรงกัน
  tradingStore.startAutoSync()
})

onUnmounted(() => {
  // หยุด Auto-Sync เมื่อออกจากหน้า
  tradingStore.stopAutoSync()
})
</script>
