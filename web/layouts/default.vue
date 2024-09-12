<template>
  <div class="navbar navbar-bottom bg-secondary text-primary-content">
    <div class="navbar-start">
    </div>
    <div class="navbar-center">
      <nuxt-link class="btn btn-ghost text-2xl" to="/">AskTube</nuxt-link>
    </div>
    <div class="navbar-end">
      <div tabindex="0" role="button" class="btn btn-ghost btn-circle avatar">
        <div class="w-10 rounded-full">
          <img  alt="AskTube" src="/images/navbar/avatar.webp" />
        </div>
      </div>

    </div>
  </div>
  <div class="container mx-auto md:px-8 px-4">
    <slot/>
  </div>
  <div class="btm-nav">
    <nuxt-link class="text-secondary" to="/discover">
      <svg
          xmlns="http://www.w3.org/2000/svg"
          class="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor">
        <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
      <span class="btm-nav-label">Explorer</span>
    </nuxt-link>
    <nuxt-link class="text-primary active" to="/">
      <svg
          xmlns="http://www.w3.org/2000/svg"
          class="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor">
        <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span class="btm-nav-label">Start</span>
    </nuxt-link>
    <nuxt-link class="text-secondary" to="/process">
      <svg
          xmlns="http://www.w3.org/2000/svg"
          class="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor">
        <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
      <span class="btm-nav-label">Processing</span>
    </nuxt-link>
  </div>
</template>
<script setup lang="js">

</script>