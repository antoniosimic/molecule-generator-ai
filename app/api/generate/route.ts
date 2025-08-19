// filepath: c:\Users\Admin\Desktop\Završni rad\mg\app\api\generate\route.ts
import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { smiles, model_id } = body // Dodajte model_id

    if (!smiles) {
      return NextResponse.json({ error: "SMILES string is required" }, { status: 400 })
    }
    if (!model_id) {
      return NextResponse.json({ error: "model_id is required" }, { status: 400 })
    }

    console.log(`API route: Calling Flask backend with SMILES: ${smiles} and model_id: ${model_id}`)

    const flaskBackendUrl = process.env.FLASK_BACKEND_URL || "http://localhost:5000/generate";
    const response = await fetch(flaskBackendUrl, { // Koristite varijablu za URL
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ smiles, model_id }), // Proslijedite model_id
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("API route: Flask backend returned error:", errorText)
      try {
         const errorJson = JSON.parse(errorText);
         return NextResponse.json({ error: errorJson.error || "Flask backend error", details: errorJson.details }, { status: response.status })
      } catch (e) {
         return NextResponse.json({ error: errorText || "Flask backend error" }, { status: response.status })
      }
    }

    const data = await response.json()
    console.log("API route: Received data from Flask backend")

    return NextResponse.json(data)
  } catch (error: any) {
    console.error("API route: Error processing request:", error)
    return NextResponse.json({ error: "Internal server error in API route", details: error.message }, { status: 500 })
  }
}