<script lang="js" setup>
import { useRoute } from "#app";
import { request, shortenWord } from "~/supports/request";
import settings from "~/supports/settings";
import ISO6391 from 'iso-639-1'
import { useLoading } from "vue-loading-overlay";
import { parseMarkdown } from '@nuxtjs/mdc/runtime'

const config = useRuntimeConfig()

const route = useRoute()
const video = ref({})
const summary = ref(null)


const providers = ref(["openai", "gemini", "claude", "mistral", "ollama"])
const aiModels = {
  "openai": ["gpt-4-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "o1-mini", "gpt-3.5-turbo"],
  "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
  "claude": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
  "mistral": ["mistral-large-latest", "open-mistral-nemo", "open-mistral-7b", "open-mixtral-8x7b", "open-mixtral-8x22b"],
  "ollama": ["qwen2", "llama3.1"]
}

const selectedSummaryProvider = ref("gemini")
const selectedSummaryModel = ref("gemini-1.5-flash")
const selectedSummaryLang = ref("en")
const summaryModels = ref(aiModels[selectedSummaryProvider.value])

const languages = ref([])
ISO6391.getAllCodes().forEach(x => {
  languages.value.unshift({
    "value": `${x}`,
    "name": `${ISO6391.getName(x)}`
  })
})


watch(selectedSummaryProvider, (newValue, oldValue) => {
  summaryModels.value = aiModels[newValue]
})

const $loading = useLoading({ ...settings.LOADING_PROPERTIES });
const doSummary = async () => {
  const loader = $loading.show({});
  try {
    console.debug(config.public.apiUrl)
    const response = await request(`${config.public.apiUrl}/api/video/summary`, 'POST', {
      "video_id": video.value.id,
      "lang_code": selectedSummaryLang.value ?? 'en',
      "provider": selectedSummaryProvider.value,
      "model": selectedSummaryModel.value
    })
    if (response.status_code >= 400) {
      useNuxtApp().$toast.error("Please select right configuration to summary", {
        autoClose: 2000
      })
    } else {
      useNuxtApp().$toast.success("Successful summary video", {
        autoClose: 2000
      })
    }
    summary.value = await parseMarkdown(response.payload)
  } finally {
    setTimeout(() => {
      loader.hide()
    }, 500)
  }

}

const selectedChatProvider = ref("gemini")
const selectedChatModel = ref("gemini-1.5-flash")
const chatModels = ref(aiModels[selectedChatProvider.value])
watch(selectedChatProvider, (newValue, oldValue) => {
  chatModels.value = aiModels[newValue]
})

const chatMessage = ref(null)
const chats = ref([])
const chatRef = ref({})

const onHoldMessageResponse = ref(false)
const onChat = async () => {
  onHoldMessageResponse.value = true
  try {
    chats.value.push({
      "question": chatMessage.value,
      "answer": null
    })
    await nextTick();
    scrollToBottom();
    const chatResponse = await request(`${config.public.apiUrl}/api/chat`, 'POST', {
      'video_id': video.value.id,
      'question': chatMessage.value,
      'provider': selectedChatProvider.value,
      'model': selectedChatModel.value
    })
    console.debug(JSON.stringify(chatResponse))
    if (chatResponse.status_code >= 400) {
      chats.value.pop()
      useNuxtApp().$toast.error("Fail to get answer from server, please try again", {
        autoClose: 2000
      })
      return
    }
    chats.value.pop()
    chats.value.push(chatResponse.payload)
    chatMessage.value = null
  } catch (e) {
    console.debug(e)
    chats.value.pop()
    useNuxtApp().$toast.error("Something went wrong, try again", {
      autoClose: 2000
    })
  } finally {
    onHoldMessageResponse.value = false
    await nextTick()
    scrollToBottom();
    chatRef.value.focus()
  }

}

const scrollToBottom = () => {
  const chatContainer = document.getElementById('chatContainer');
  if (chatContainer) {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
};

const doClearChat = async () => {
  try {
    await request(`${config.public.apiUrl}/api/chat/clear/${route.params.id}`, 'DELETE')
    chats.value = []
  } catch (e) {
    useNuxtApp().$toast.error("Something went wrong, try later", {
      autoClose: 2000
    })
  }

}


onMounted(async () => {
  try {
    const videoResponse = await request(`${config.public.apiUrl}/api/video/detail/${route.params.id}`, 'GET')
    video.value = videoResponse.payload
    summary.value = videoResponse.payload.summary ? await parseMarkdown(videoResponse.payload.summary) : null

    const chatResponse = await request(`${config.public.apiUrl}/api/chat/history/${route.params.id}`, 'GET')
    chats.value = chatResponse.payload
    await nextTick();
    if (chats.value.length > 0) {
      scrollToBottom();
    }
    chatRef.value.focus()
  } catch (e) {
    useNuxtApp().$toast.error("Something went wrong, try later", {
      autoClose: 2000
    })
  }
})

</script>

<template>
  <div class="flex flex-col md:flex-row">
    <div class="basis-2/5 border-amber-300 w-full h-screen">
      <div class="card bg-base-100 shadow-xl w-full ">
        <figure class="w-full">
          <iframe :src="video.play_url" class="w-full rounded-lg shadow-2xl" height="400" />
        </figure>
        <div class="card-body rounded-2xl">
          <h1 class="card-title"> {{ shortenWord(video.title) }}</h1>
          <div class="collapse collapse-arrow bg-base-300">
            <input type="checkbox" />
            <div class="collapse-title text-xl font-medium rounded-2xl">Summary Settings</div>
            <div class="collapse-content">
              <div class="flex flex-col lg:flex-row ">
                <div class="md:basis-1/3 p-2">
                  <select v-model="selectedSummaryProvider" class="select select-bordered  w-full">
                    <option disabled selected>Provider</option>
                    <option v-for="item in providers" :value="item">{{ item }}</option>
                  </select>
                </div>
                <div class="md:basis-1/3 p-2">
                  <select v-model="selectedSummaryModel" class="select select-bordered  w-full">
                    <option disabled selected>Model</option>
                    <option v-for="model in summaryModels" :value="model">{{ model }}</option>
                  </select>
                </div>
                <div class="md:basis-1/3 p-2">
                  <select v-model="selectedSummaryLang" class="select select-bordered  w-full">
                    <option disabled selected>Language</option>
                    <option v-for="lang in languages" :value="lang.value"> {{ lang.name }}</option>
                  </select>
                </div>
              </div>
              <div class="flex justify-end p-2">
                <button class="btn btn-primary " @click="doSummary">Summary</button>
              </div>
            </div>
          </div>
          <div class="divider">🌟</div>
          <div class="collapse collapse-arrow bg-base-200 md:hidden">
            <input type="checkbox" />
            <div class="collapse-title text-xl font-medium rounded-2xl">Read Summary</div>
            <div class="collapse-content">
              <article v-if="summary" class="prose overflow-auto h-96">
                <MDCRenderer :body="summary.body" :data="summary.data" />
              </article>
            </div>
          </div>
          <div class="hidden md:block">
            <article v-if="summary" class="prose overflow-auto h-96">
              <MDCRenderer :body="summary.body" :data="summary.data" />
            </article>
          </div>


        </div>
      </div>
    </div>
    <div class="divider lg:divider-horizontal">🌟</div>
    <div class="basis-3/5">
      <div class="h-screen flex items-start justify-center border-amber-300">
        <div class="card w-full h-3/4 shadow-xl flex flex-col">
          <div class="card-header p-4">
            <div class="collapse collapse-arrow bg-base-300">
              <input type="checkbox" />
              <div class="collapse-title text-xl font-medium rounded-2xl">Chat Settings</div>
              <div class="collapse-content">
                <div class="flex flex-col lg:flex-row ">
                  <div class="md:basis-1/2 p-2">
                    <select v-model="selectedChatProvider" class="select select-bordered  w-full">
                      <option disabled selected>Provider</option>
                      <option v-for="item in providers" :value="item">{{ item }}</option>
                    </select>
                  </div>
                  <div class="md:basis-1/2 p-2">
                    <select v-model="selectedChatModel" class="select select-bordered  w-full">
                      <option disabled selected>Model</option>
                      <option v-for="model in chatModels" :value="model">{{ model }}</option>
                    </select>
                  </div>
                </div>
                <div class="flex justify-end p-2">
                  <button class="btn btn-primary " @click="doClearChat">Clear Messages</button>
                </div>
              </div>

            </div>
          </div>
          <div class="card-body flex-grow flex flex-col overflow-hidden p-4">
            <div id="chatContainer" class="flex-grow overflow-y-auto space-y-4 rounded-2xl p-4 mb-4">
              <div v-for="chat in chats">
                <div class="chat chat-end">
                  <div class="chat-bubble bg-base-300 text-base-content">{{ chat.question }}</div>
                </div>
                <div v-if="chat.answer" class="chat chat-start">
                  <div class="chat-bubble bg-base-100 text-base-content">
                    <MDC :value="chat.answer" tag="article" />
                  </div>
                  <div class="chat-footer opacity-90 mt-1 ml-2">
                    <div class="badge badge-primary gap-2">{{ chat.provider }}</div>
                  </div>
                </div>
                <div v-else class="chat chat-start">
                  <button class="btn btn-square btn-ghost">
                    <span class="loading loading-spinner"></span>
                  </button>
                </div>
              </div>

            </div>

            <div class="flex mt-auto">
              <input ref="chatRef" v-model="chatMessage" :disabled="onHoldMessageResponse"
                class="input input-bordered flex-grow " placeholder="Type your message here..." type="text"
                @keydown.enter="onChat" />
              <button :disabled="onHoldMessageResponse" class="ml-5 btn btn-primary " @click="onChat">Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped></style>