const API_URL =
  "http://127.0.0.1:8000";


export async function predictAudio(
  formData: FormData
) {

  const response = await fetch(
    `${API_URL}/predict`,
    {
      method: "POST",
      body: formData,
    }
  );

  return response.json();
}


export async function getStats() {

  const response = await fetch(
    `${API_URL}/dashboard/stats`
  );

  return response.json();
}