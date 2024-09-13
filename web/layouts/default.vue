<template>
  <div class="navbar navbar-bottom bg-secondary text-primary-content">
    <div class="navbar-start">
    </div>
    <div class="navbar-center">
      <nuxt-link class="btn btn-ghost text-2xl" to="/">AskTube</nuxt-link>
    </div>
    <div class="navbar-end">
      <div class="btn btn-ghost btn-circle avatar" role="button" tabindex="0">
        <div class="w-10 rounded-full">
          <img alt="AskTube" src="/images/navbar/avatar.webp"/>
        </div>
      </div>

    </div>
  </div>
  <div class="container mx-auto md:px-8 px-4">
    <slot/>
  </div>
  <div class="btm-nav">
    <nuxt-link :class="discoverStyle" to="/discover">
      <i class="fa-solid fa-chart-simple"></i>
      <span class="btm-nav-label">Explorer</span>
    </nuxt-link>
    <nuxt-link :class="indexStyle" to="/">
      <i class="fa-solid fa-rocket"></i>
      <span class="btm-nav-label">Start</span>
    </nuxt-link>
    <nuxt-link :class="processStyle" to="/process">
      <i class="fa-solid fa-spinner"></i>
      <span class="btm-nav-label">Process</span>
    </nuxt-link>
  </div>
</template>
<script lang="js" setup>
const route = useRoute()
const discoverStyle = ref("text-secondary")
const indexStyle = ref("text-secondary")
const processStyle = ref("text-secondary")

onBeforeMount(() => {
  if (route.fullPath === "/") {
    indexStyle.value = "text-primary active"
  }
  if (route.fullPath === "/discover") {
    discoverStyle.value = "text-primary active"
  }
  if (route.fullPath === "/process") {
    processStyle.value = "text-primary active"
  }
})

watch(() => route.fullPath, (newPath, oldPath) => {
      if (newPath === "/") {
        indexStyle.value = "text-primary active"
        processStyle.value = "text-secondary"
        discoverStyle.value = "text-secondary"
      }
      if (newPath === "/discover") {
        indexStyle.value = "text-secondary"
        processStyle.value = "text-secondary"
        discoverStyle.value = "text-primary active"
      }
      if (newPath === "/process") {
        indexStyle.value = "text-secondary"
        processStyle.value = "text-primary active"
        discoverStyle.value = "text-secondary"
      }
    }
);
</script>