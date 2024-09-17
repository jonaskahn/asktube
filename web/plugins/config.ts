export default defineNuxtPlugin((nuxtApp) => {
    const config = useRuntimeConfig()

    console.debug('API base URL:', config.public.apiUrl)
});
