import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../services/api'

export const useSignalStore = defineStore('signal', () => {
  // State
  const currentSignal = ref(null)
  const signalHistory = ref([])
  const matchedPatterns = ref([])
  const isLoading = ref(false)
  const error = ref(null)
  const selectedSymbol = ref('EURUSD')
  const selectedTimeframe = ref('H1')

  // Getters
  const signalClass = computed(() => {
    if (!currentSignal.value) return 'signal-wait'
    const signal = currentSignal.value.signal.toLowerCase().replace('_', '-')
    return `signal-${signal}`
  })

  const isBullish = computed(() => {
    if (!currentSignal.value) return false
    return ['STRONG_BUY', 'BUY'].includes(currentSignal.value.signal)
  })

  const isBearish = computed(() => {
    if (!currentSignal.value) return false
    return ['STRONG_SELL', 'SELL'].includes(currentSignal.value.signal)
  })

  const shouldTrade = computed(() => {
    if (!currentSignal.value) return false
    return currentSignal.value.signal !== 'WAIT' && currentSignal.value.confidence >= 70
  })

  // Actions
  async function analyzePattern(pattern, currentPrice) {
    isLoading.value = true
    error.value = null

    try {
      const response = await api.analyze({
        symbol: selectedSymbol.value,
        timeframe: selectedTimeframe.value,
        current_pattern: pattern,
        current_price: currentPrice,
        k: 10
      })

      currentSignal.value = response
      signalHistory.value.unshift(response)

      // Keep only last 100 signals
      if (signalHistory.value.length > 100) {
        signalHistory.value.pop()
      }

      return response
    } catch (err) {
      error.value = err.message || 'Failed to analyze pattern'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function fetchSampleSignal() {
    isLoading.value = true
    error.value = null

    try {
      const response = await api.generateSampleSignal(
        selectedSymbol.value,
        selectedTimeframe.value
      )
      currentSignal.value = response
      return response
    } catch (err) {
      error.value = err.message
      throw err
    } finally {
      isLoading.value = false
    }
  }

  function setSymbol(symbol) {
    selectedSymbol.value = symbol
  }

  function setTimeframe(timeframe) {
    selectedTimeframe.value = timeframe
  }

  function clearSignal() {
    currentSignal.value = null
    error.value = null
  }

  return {
    // State
    currentSignal,
    signalHistory,
    matchedPatterns,
    isLoading,
    error,
    selectedSymbol,
    selectedTimeframe,

    // Getters
    signalClass,
    isBullish,
    isBearish,
    shouldTrade,

    // Actions
    analyzePattern,
    fetchSampleSignal,
    setSymbol,
    setTimeframe,
    clearSignal
  }
})
