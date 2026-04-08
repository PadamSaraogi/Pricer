const CACHE_NAME = 'stlite-cache-v2';
const ASSETS_TO_CACHE = [
  '/',
  'https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.js',
  'https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.css',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Cache heavy assets from CDNs (Pyodide, Wheels)
  if (url.hostname === 'cdn.jsdelivr.net' || url.hostname === 'files.pythonhosted.org') {
    event.respondWith(
      caches.match(event.request).then((response) => {
        if (response) return response;
        return fetch(event.request).then((networkResponse) => {
          if (!networkResponse || networkResponse.status !== 200) return networkResponse;
          return caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          });
        });
      })
    );
    return;
  }

  // Standard strategy for other requests
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
