<script lang="js" setup>
import settings from "~/supports/settings.js";
import {useLoading} from "vue-loading-overlay";
import {request, shortenWord} from "~/supports/request.js";

const config = useRuntimeConfig()

const videos = ref([])
const total = ref(0)
const pageNo = ref(1)

const fetchVideos = async () => {
  console.debug(config.public.apiUrl)
  const response = await request(`${config.public.apiUrl}/api/videos/${pageNo.value}`, 'GET')
  if (response.status_code >= 400) {
    return
  }
  total.value = response.payload.total
  videos.value.push(...response.payload.videos)
}

const loadMore = async () => {
  if (videos.value.length >= total.value) {
    return
  }
  try {
    pageNo.value++
    await fetchVideos()
  } catch (e) {
    pageNo.value--
    useNuxtApp().$toast.error("Cannot load more records", {
      autoClose: 2000
    })
  }
}

onBeforeMount(async () => {
  await fetchVideos()
})


const url = ref(null)
const invalidUrl = ref(false)
const provider = ref('Analysis Provider')
const invalidProvider = ref(false)


const onOpenAddDialog = () => {
  document.getElementById('chatModal').showModal()
}
const onCloseAddDialog = () => {
  document.getElementById('chatModal').close()
}

const displayLoadMore = computed(() => videos.value.length > total.value)

const $loading = useLoading({...settings.LOADING_PROPERTIES});
const process = async () => {
  const error = []
  if (!url.value) {
    error.push('Enter a valid YouTube URL')
    invalidUrl.value = true
  }

  if (!provider.value || provider.value === 'Provider') {
    error.push('Select a valid Provider')
    invalidProvider.value = true
  }

  if (error.length > 0) {
    console.debug("Please enter valid youtube url and select right provider")
    return
  }

  invalidUrl.value = false
  invalidProvider.value = false
  onCloseAddDialog()
  const loader = $loading.show({});
  try {
    const data = await request(`${config.public.apiUrl}/api/youtube/process`, 'POST', {
      url: url.value,
      provider: provider.value
    })
    if (data.status_code >= 400) {
      useNuxtApp().$toast.error("Please provide valid Youtube URL and Provider", {
        autoClose: 2000
      })
    } else {
      useNuxtApp().$toast.success("Successful fetching new Youtube Video", {
        autoClose: 2000
      })
      url.value = null
      videos.value.unshift(data.payload)
      if (videos.value >= 50) {
        videos.value.pop()
      }
    }
    // try to fetch analysis immediately
   try {
     request(config.public.apiUrl + `/api/video/analysis`, 'POST', {
        'video_id': data.payload.id
      })
   } catch (error) {
      console.warn(error)
   }
  } catch (e) {
    console.debug(e)
  } finally {
    setTimeout(() => {
      loader.hide()
    }, 500)
  }
}

const analysis = async (id) => {
  const data = await request(config.public.apiUrl + `/api/video/analysis`, 'POST', {
    'video_id': id
  })
  if (data.status_code >= 400) {
    useNuxtApp().$toast.error("Something went wrong", {
      autoClose: 2000
    })
  } else {
    const payload = data.payload
    useNuxtApp().$toast.success(payload.analysis_state === 1 ? "Video is ready" : "Video is in analysis process", {
      autoClose: 2000
    })
    const index = videos.value.findIndex(item => item.id === payload.id);
    if (index !== -1) {
      videos.value.splice(index, 1);
    }
    videos.value.splice(index, 0, payload);
  }
}
</script>

<template>
  <div class="grid grid-cols-1 gap-1 sm:grid-cols-2 sm:gap-2  lg:grid-cols-3 lg:gap-3 xl:grid-cols-4 xl:gap-4">
    <div v-for="item in videos" class="card bg-base-100 w-full shadow-xl">
      <figure>
        <img :src="item.thumbnail" style="height: 200px; width: 300px"/>
      </figure>
      <div class="card-body">
        <h2 :data-tip="item.title" class="card-title tooltip">{{ shortenWord(item.title) }}</h2>

        <p :data-tip="item.description" class="tooltip">{{ shortenWord(item.description) }}</p>
        <div class="card-actions justify-end">
          <a v-show="item.analysis_state === 1" :href="`/chat/${item.id}`" class="btn btn-primary" target="_blank">
            Chat now
          </a>
          <button v-show="item.analysis_state === 0" class="btn btn-secondary text-base-100" @click="analysis(item.id)">
            Analysis
          </button>
          <button v-show="item.analysis_state === 2" class="btn btn-secondary text-base-100" @click="analysis(item.id)">
            Refresh
          </button>
        </div>
      </div>
    </div>
  </div>
  <button class="fixed bottom-4 left-4 transform -translate-x-1 btn btn-primary px-8 py-4 rounded-full shadow-lg "
          @click="onOpenAddDialog">
    Add
  </button>
  <button v-if="displayLoadMore"
          class="fixed bottom-4 right-4  transform -translate-x-1 btn btn-accent  px-8 py-4 rounded-full shadow-lg"
          @click="loadMore">
    Load More
  </button>

  <dialog id="chatModal" class="modal">
    <div class="modal-box">
      <form method="dialog">
        <button class="btn btn-sm btn-circle btn-ghost absolute right-2 top-2">✕</button>
      </form>
      <h3 class="text-lg font-bold mb-5">Add new video ✨</h3>
      <input v-model="url" :class="{ 'input-error': invalidUrl }" class="input input-bordered w-full rounded-full"
             placeholder="Enter Youtube URL . . ."/>
      <div class="modal-action align-middle flex justify-between">
        <div class="justify-start">
          <select v-model="provider" :class="{ 'select-error': invalidProvider }"
                  class="select select-bordered w-full rounded-full">
            <option disabled selected>Analysis Provider</option>
            <option>local</option>
            <option>gemini</option>
            <option>voyageai</option>
            <option>openai</option>
            <option>mistral</option>
          </select>
        </div>
        <div class="justify-end">
          <button class="btn btn-secondary text-base-100 rounded-full" @click="process">Process Now</button>
        </div>
      </div>
    </div>
  </dialog>
</template>

<style scoped></style>