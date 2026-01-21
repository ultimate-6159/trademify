<template>
  <div class="h-64">
    <v-chart :option="chartOption" autoresize class="h-full w-full" />
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  voteDetails: {
    type: Object,
    default: () => ({ bullish: 0, bearish: 0, total: 0 })
  }
})

const chartOption = computed(() => {
  const bullish = props.voteDetails.bullish || 0
  const bearish = props.voteDetails.bearish || 0
  const total = bullish + bearish || 1
  
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: '#1E293B',
      borderColor: '#334155',
      textStyle: {
        color: '#F8FAFC'
      },
      formatter: '{b}: {c} ({d}%)'
    },
    series: [
      {
        name: 'Votes',
        type: 'pie',
        radius: ['50%', '70%'],
        center: ['50%', '50%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 4,
          borderColor: '#0F172A',
          borderWidth: 2
        },
        label: {
          show: true,
          position: 'center',
          formatter: () => {
            const winningVotes = Math.max(bullish, bearish)
            const percentage = Math.round((winningVotes / total) * 100)
            return `{value|${percentage}%}\n{label|Confidence}`
          },
          rich: {
            value: {
              fontSize: 28,
              fontWeight: 'bold',
              color: '#F8FAFC'
            },
            label: {
              fontSize: 12,
              color: '#94A3B8',
              padding: [5, 0, 0, 0]
            }
          }
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 16,
            fontWeight: 'bold'
          }
        },
        labelLine: {
          show: false
        },
        data: [
          {
            value: bullish,
            name: 'Bullish',
            itemStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  { offset: 0, color: '#22C55E' },
                  { offset: 1, color: '#10B981' }
                ]
              }
            }
          },
          {
            value: bearish,
            name: 'Bearish',
            itemStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  { offset: 0, color: '#EF4444' },
                  { offset: 1, color: '#DC2626' }
                ]
              }
            }
          }
        ]
      }
    ]
  }
})
</script>
