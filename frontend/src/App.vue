<template>
  <div class="min-h-screen bg-dark-300 flex flex-col">
    <!-- Header -->
    <header class="bg-dark-200 border-b border-gray-800 sticky top-0 z-50">
      <div class="container mx-auto px-3 sm:px-4 py-3 sm:py-4">
        <div class="flex items-center justify-between">
          <!-- Logo -->
          <div class="flex items-center space-x-2 sm:space-x-4">
            <h1 class="text-xl sm:text-2xl font-bold text-white">
              <span class="text-primary">🔮</span>
              <span class="hidden xs:inline">Trade</span>mify
            </h1>
            <span class="hidden sm:inline text-xs sm:text-sm text-gray-500"
              >AI Pattern Recognition</span
            >
          </div>

          <!-- Desktop Navigation -->
          <nav class="hidden md:flex items-center space-x-4 lg:space-x-6">
            <router-link
              to="/"
              class="text-gray-400 hover:text-white transition-colors flex items-center gap-1 text-sm lg:text-base"
              :class="{ 'text-white': $route.path === '/' }"
            >
              <span>🔬</span> Control Center
            </router-link>
            <router-link
              to="/intelligence"
              class="text-gray-400 hover:text-white transition-colors flex items-center gap-1 text-sm lg:text-base"
              :class="{ 'text-white': $route.path === '/intelligence' }"
            >
              <span>🧠</span> Intelligence
            </router-link>
            <router-link
              to="/settings"
              class="text-gray-400 hover:text-white transition-colors flex items-center gap-1 text-sm lg:text-base"
              :class="{ 'text-white': $route.path === '/settings' }"
            >
              <span>⚙️</span> Settings
            </router-link>
          </nav>

          <!-- Mobile Menu Button -->
          <button
            @click="mobileMenuOpen = !mobileMenuOpen"
            class="md:hidden p-2 rounded-lg bg-gray-800 text-white"
          >
            <span v-if="!mobileMenuOpen">☰</span>
            <span v-else>✕</span>
          </button>
        </div>

        <!-- Mobile Navigation -->
        <nav
          v-if="mobileMenuOpen"
          class="md:hidden mt-3 pt-3 border-t border-gray-700 space-y-2"
        >
          <router-link
            to="/"
            @click="mobileMenuOpen = false"
            class="block py-2 px-3 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
            :class="{ 'text-white bg-gray-800': $route.path === '/' }"
          >
            <span>🔬</span> Control Center
          </router-link>
          <router-link
            to="/intelligence"
            @click="mobileMenuOpen = false"
            class="block py-2 px-3 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
            :class="{
              'text-white bg-gray-800': $route.path === '/intelligence',
            }"
          >
            <span>🧠</span> Intelligence (20 Layers)
          </router-link>
          <router-link
            to="/settings"
            @click="mobileMenuOpen = false"
            class="block py-2 px-3 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
            :class="{ 'text-white bg-gray-800': $route.path === '/settings' }"
          >
            <span>⚙️</span> Settings
          </router-link>
        </nav>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 container mx-auto px-2 sm:px-4 py-3 sm:py-6">
      <router-view />
    </main>

    <!-- Mobile Bottom Navigation -->
    <nav
      class="md:hidden fixed bottom-0 left-0 right-0 bg-dark-200 border-t border-gray-800 z-50"
    >
      <div class="flex justify-around py-2">
        <router-link
          to="/"
          class="flex flex-col items-center py-1 px-3 text-xs"
          :class="$route.path === '/' ? 'text-blue-400' : 'text-gray-400'"
        >
          <span class="text-lg">🔬</span>
          <span>Control</span>
        </router-link>
        <router-link
          to="/intelligence"
          class="flex flex-col items-center py-1 px-3 text-xs"
          :class="
            $route.path === '/intelligence' ? 'text-blue-400' : 'text-gray-400'
          "
        >
          <span class="text-lg">🧠</span>
          <span>AI</span>
        </router-link>
        <router-link
          to="/settings"
          class="flex flex-col items-center py-1 px-3 text-xs"
          :class="
            $route.path === '/settings' ? 'text-blue-400' : 'text-gray-400'
          "
        >
          <span class="text-lg">⚙️</span>
          <span>Settings</span>
        </router-link>
      </div>
    </nav>

    <!-- Footer (Desktop only) -->
    <footer class="hidden md:block bg-dark-200 border-t border-gray-800 py-4">
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

    <!-- Spacer for mobile bottom nav -->
    <div class="md:hidden h-16"></div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";

const apiStatus = ref(null);
const mobileMenuOpen = ref(false);

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
