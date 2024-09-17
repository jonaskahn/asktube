// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
    compatibilityDate: '2024-04-03',
    devtools: {enabled: true},
    modules: ['@nuxtjs/tailwindcss', '@nuxtjs/mdc'],
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
    },
    css: ['vue-loading-overlay/dist/css/index.css', 'assets/css/global.css'],
    mdc: {
        components: {
            prose: false,
            map: {
                h1: 'h3',
                h2: 'h4',
                h3: 'strong',
                h4: 'strong',
                h5: 'strong',
                h6: 'strong'
            }
        }
    },
    runtimeConfig: {
        public: {
            apiUrl: 'http://localhost:8000/api',
        }
    },
})