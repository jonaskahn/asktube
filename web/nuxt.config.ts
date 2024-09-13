// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
    compatibilityDate: '2024-04-03',
    devtools: {enabled: true},
    modules: ['@nuxtjs/tailwindcss', '@vesp/nuxt-fontawesome'],
    app: {
        pageTransition: {name: 'slide-in', mode: 'out-in'},
        head: {
            script: [
                {
                    src: 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/js/all.min.js',
                    defer: true
                }
            ]
        }
    }
})