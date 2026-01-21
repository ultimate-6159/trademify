import { createRouter, createWebHistory } from "vue-router";
import Dashboard from "../views/Dashboard.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "dashboard",
      component: Dashboard,
    },
    {
      path: "/analysis/:symbol?",
      name: "analysis",
      component: () => import("../views/Analysis.vue"),
    },
    {
      path: "/enhanced",
      name: "enhanced",
      component: () => import("../views/EnhancedAnalysis.vue"),
    },
    {
      path: "/intelligence",
      name: "intelligence",
      component: () => import("../views/Intelligence.vue"),
    },
    {
      path: "/trading",
      name: "trading",
      component: () => import("../views/Trading.vue"),
    },
    {
      path: "/settings",
      name: "settings",
      component: () => import("../views/Settings.vue"),
    },
  ],
});

export default router;
