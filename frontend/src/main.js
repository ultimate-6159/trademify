import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import './assets/main.css'

// Firebase real-time sync
import { initFirebase } from './services/firebase'

// ECharts
import ECharts from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, CandlestickChart, PieChart, BarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent,
  MarkLineComponent,
  MarkAreaComponent
} from 'echarts/components'

// Register ECharts components
use([
  CanvasRenderer,
  LineChart,
  CandlestickChart,
  PieChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent,
  MarkLineComponent,
  MarkAreaComponent
])

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.component('v-chart', ECharts)

// Initialize Firebase for real-time sync (if configured)
const firebaseInitialized = initFirebase()
if (firebaseInitialized) {
  console.log('[App] Firebase real-time sync enabled')
} else {
  console.log('[App] Firebase not configured - using API polling')
}

app.mount('#app')
