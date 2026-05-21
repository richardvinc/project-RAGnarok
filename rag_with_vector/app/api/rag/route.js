const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request) {
  try {
    const payload = await request.json();
    const response = await fetch(`${BACKEND_URL}/rag`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    return new Response(text, {
      status: response.status,
      headers: {
        "Content-Type":
          response.headers.get("content-type") ?? "application/json",
      },
    });
  } catch (error) {
    return Response.json(
      {
        detail:
          error instanceof Error ? error.message : "Failed to reach backend",
      },
      { status: 500 },
    );
  }
}
