import { createRouter, createWebHistory } from "vue-router";
import PipelineDashboard from "../views/PipelineDashboard.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "control-center",
      component: PipelineDashboard,
    },
    {
      path: "/realtime",
      name: "realtime",
      component: () => import("../views/RealTimeDashboard.vue"),
    },
    {
      path: "/settings",
      name: "settings",
      component: () => import("../views/Settings.vue"),
    },
    // Legacy routes - redirect to main
    {
      path: "/pipeline",
      redirect: "/",
    },
    {
      path: "/dashboard",
      redirect: "/realtime",
    },
    {
      path: "/trading",
      redirect: "/",
    },
    {
      path: "/enhanced",
      redirect: "/",
    },
    {
      path: "/intelligence",
      redirect: "/",
    },
    {
      path: "/analysis/:symbol?",
      redirect: "/",
    },
  ],
});

export default router;
