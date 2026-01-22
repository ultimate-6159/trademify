<template>
  <div class="min-h-screen bg-dark-300">
    <!-- Header -->
    <header class="bg-dark-200 border-b border-gray-800">
      <div class="container mx-auto px-4 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-4">
            <h1 class="text-2xl font-bold text-white">
              <span class="text-primary">Trade</span>mify
            </h1>
            <span class="text-sm text-gray-500">AI Pattern Recognition</span>
          </div>

          <nav class="flex items-center space-x-6">
            <router-link
              to="/"
              class="text-gray-400 hover:text-white transition-colors flex items-center gap-1"
              :class="{ 'text-white': $route.path === '/' }"
            >
              <span>🔬</span> Control Center
            </router-link>
            <router-link
              to="/settings"
              class="text-gray-400 hover:text-white transition-colors"
              :class="{ 'text-white': $route.path === '/settings' }"
            >
              Settings
            </router-link>
          </nav>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto px-4 py-6">
      <router-view />
    </main>

    <!-- Footer -->
    <footer class="bg-dark-200 border-t border-gray-800 py-4 mt-auto">
      <div class="container mx-auto px-4 text-center text-gray-500 text-sm">
        Trademify © 2024 - AI-Powered Pattern Recognition Trading System
        <span
          v-if="apiStatus"
          class="ml-2 px-2 py-0.5 rounded text-xs"
          :class="
            apiStatus === 'online'
              ? 'bg-green-900 text-green-400'
              : 'bg-yellow-900 text-yellow-400'
          "
        >
          API: {{ apiStatus }}
        </span>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";

const apiStatus = ref(null);

// Auto-detect API URL (returns base WITH /api/v1)
function getApiBase() {
  const hostname = window.location.hostname;
  if (hostname !== "localhost" && hostname !== "127.0.0.1") {
    return `http://${hostname}:8000/api/v1`;
  }
  const envUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
  // Ensure /api/v1 suffix
  return envUrl.includes("/api/v1") ? envUrl : `${envUrl}/api/v1`;
}

onMounted(async () => {
  const apiBase = getApiBase();

  // Direct fetch without mock fallback
  try {
    const response = await fetch(`${apiBase}/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    if (response.ok) {
      const data = await response.json();
      console.log("[App] API Health:", data);
      apiStatus.value = "online";
    } else {
      console.warn("[App] API returned:", response.status);
      apiStatus.value = "error";
    }
  } catch (error) {
    console.warn("[App] API unreachable:", error.message);
    apiStatus.value = "offline";
  }
});
</script>
