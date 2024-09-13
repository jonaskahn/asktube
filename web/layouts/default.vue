<script lang="js" setup>
const route = useRoute()
const discoverStyle = ref("text-secondary")
const indexStyle = ref("text-secondary")
const processStyle = ref("text-secondary")

onMounted(() => {
  if (route.fullPath === "/") {
    indexStyle.value = "text-primary active bg-base-300"
  }
  if (route.fullPath === "/discover") {
    discoverStyle.value = "text-primary active bg-base-300"
  }
  if (route.fullPath === "/process") {
    processStyle.value = "text-primary active bg-base-300"
  }
})

watch(() => route.fullPath, (newPath, oldPath) => {
      if (newPath === "/") {
        indexStyle.value = "text-primary active bg-base-300"
        processStyle.value = "text-secondary"
        discoverStyle.value = "text-secondary"
      }
      if (newPath === "/discover") {
        indexStyle.value = "text-secondary"
        processStyle.value = "text-secondary"
        discoverStyle.value = "text-primary active bg-base-300"
      }
      if (newPath === "/process") {
        indexStyle.value = "text-secondary"
        processStyle.value = "text-primary active bg-base-300"
        discoverStyle.value = "text-secondary"
      }
    }
);
</script>
<template>
  <div class="navbar navbar-bottom bg-secondary text-base-100">
    <div class="navbar-start">

    </div>
    <div class="navbar-center">
      <nuxt-link class="btn btn-ghost text-2xl" to="/">AskTube</nuxt-link>
    </div>
    <div class="navbar-end">
      <div class="btn btn-ghost btn-circle avatar" role="button" tabindex="0">
        <div class="w-10 rounded-full">
          <nuxt-link target="_blank" to="https://github.com/jonaskahn/asktube">
            <img alt="AskTube" src="https://avatars.githubusercontent.com/u/4338500"/>
          </nuxt-link>
        </div>
      </div>

    </div>
  </div>
  <div class="container mx-auto md:px-8 px-4 pb-20">
    <slot/>
  </div>
</template>
