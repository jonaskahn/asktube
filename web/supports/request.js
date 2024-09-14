const request = async (url, method, body) => {
  return await $fetch(url, {
    method: method,
    body: body
  })
    .then((response) => response)
    .catch((error) => error.data)
}

const shortenWord = (str) => {
  if (str === null || str === undefined) return str;
  if (str.length <= 200) return str;
  return str.slice(0, 200) + "...";
}


export {shortenWord, request}