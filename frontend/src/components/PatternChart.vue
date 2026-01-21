<template>
  <div class="h-full w-full">
    <v-chart :option="chartOption" autoresize class="h-full w-full" />
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  currentPattern: {
    type: Array,
    default: () => []
  },
  matchedPatterns: {
    type: Array,
    default: () => []
  },
  projectedMovement: {
    type: Array,
    default: () => []
  }
})

const chartOption = computed(() => {
  // Generate x-axis data
  const patternLength = props.currentPattern.length || 60
  const futureLength = props.projectedMovement.length || 10
  const totalLength = patternLength + futureLength
  
  const xAxisData = []
  for (let i = 0; i < totalLength; i++) {
    if (i < patternLength) {
      xAxisData.push(`-${patternLength - i}`)
    } else {
      xAxisData.push(`+${i - patternLength + 1}`)
    }
  }
  
  // Current pattern series (solid line)
  const currentData = [...props.currentPattern]
  // Pad with null for future
  for (let i = 0; i < futureLength; i++) {
    currentData.push(null)
  }
  
  // Projected movement (extend current pattern)
  const projectedData = new Array(patternLength).fill(null)
  if (props.projectedMovement.length > 0 && props.currentPattern.length > 0) {
    const lastPrice = props.currentPattern[props.currentPattern.length - 1]
    props.projectedMovement.forEach((change) => {
      projectedData.push(lastPrice * (1 + change / 100))
    })
  }
  
  // Matched patterns series (faded lines)
  const matchedSeries = props.matchedPatterns.map((pattern, idx) => {
    const data = [...pattern.data]
    // Extend with future projection if available
    if (pattern.future) {
      pattern.future.forEach(p => data.push(p))
    } else {
      for (let i = 0; i < futureLength; i++) {
        data.push(null)
      }
    }
    
    return {
      name: `Pattern ${idx + 1}`,
      type: 'line',
      data: data,
      smooth: true,
      symbol: 'none',
      lineStyle: {
        width: 1,
        opacity: 0.3,
        color: '#64748B'
      }
    }
  })
  
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1E293B',
      borderColor: '#334155',
      textStyle: {
        color: '#F8FAFC'
      }
    },
    legend: {
      show: false
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '10%',
      top: '5%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: xAxisData,
      axisLine: {
        lineStyle: {
          color: '#334155'
        }
      },
      axisLabel: {
        color: '#94A3B8'
      },
      splitLine: {
        show: false
      }
    },
    yAxis: {
      type: 'value',
      axisLine: {
        lineStyle: {
          color: '#334155'
        }
      },
      axisLabel: {
        color: '#94A3B8'
      },
      splitLine: {
        lineStyle: {
          color: '#1E293B'
        }
      }
    },
    series: [
      // Current pattern (main solid line)
      {
        name: 'Current',
        type: 'line',
        data: currentData,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 3,
          color: '#3B82F6'
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0)' }
            ]
          }
        }
      },
      // Projected movement
      {
        name: 'Projected',
        type: 'line',
        data: projectedData,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          type: 'dashed',
          color: '#10B981'
        }
      },
      // Add vertical line at current position
      {
        name: 'Now',
        type: 'line',
        markLine: {
          symbol: 'none',
          data: [
            {
              xAxis: patternLength - 1,
              lineStyle: {
                color: '#F59E0B',
                type: 'dashed',
                width: 2
              },
              label: {
                formatter: 'NOW',
                color: '#F59E0B'
              }
            }
          ]
        }
      },
      // Matched patterns
      ...matchedSeries
    ],
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100
      }
    ]
  }
})
</script>
