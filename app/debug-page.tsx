"use client"

import { useState, useEffect, useRef } from "react"
import axios from "axios"
import MoleculeLoadingAnimation from "@/components/molecule-loading-animation"

// This is a simplified version of the page for debugging
export default function DebugPage() {
  const [inputSmiles, setInputSmiles] = useState("CCO") // Default to ethanol for testing
  const [molecule, setMolecule] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const inputViewerRef = useRef(null)
  const generatedViewerRef = useRef(null)

  const generateMolecule = () => {
    setLoading(true)
    setError("")
    console.log("Generating molecule for SMILES:", inputSmiles)

    axios
      .post("http://localhost:5000/generate", { smiles: inputSmiles })
      .then((response) => {
        console.log("Received data:", response.data)
        setMolecule(response.data)
        setLoading(false)
      })
      .catch((error) => {
        console.error("Error:", error)
        setError(error.toString())
        setLoading(false)
      })
  }

  // Initialize 3Dmol.js viewers when new 3D structure data is available
  useEffect(() => {
    if (molecule && window.$3Dmol) {
      console.log("Setting up 3D viewers")

      if (molecule.input_mol_block && inputViewerRef.current) {
        inputViewerRef.current.innerHTML = ""
        const config = { backgroundColor: "0xffffff" }
        const viewer = window.$3Dmol.createViewer(inputViewerRef.current, config)
        viewer.addModel(molecule.input_mol_block, "mol")
        viewer.setStyle(
          {},
          {
            stick: { radius: 0.15, colorscheme: "jmol" },
            sphere: { radius: 0.5, colorscheme: "jmol" },
          },
        )
        viewer.zoomTo()
        viewer.render()
      }

      if (molecule.generated_mol_block && generatedViewerRef.current) {
        generatedViewerRef.current.innerHTML = ""
        const config = { backgroundColor: "0xffffff" }
        const viewer = window.$3Dmol.createViewer(generatedViewerRef.current, config)
        viewer.addModel(molecule.generated_mol_block, "mol")
        viewer.setStyle(
          {},
          {
            stick: { radius: 0.15, colorscheme: "jmol" },
            sphere: { radius: 0.5, colorscheme: "jmol" },
          },
        )
        viewer.zoomTo()
        viewer.render()
      }
    }
  }, [molecule])

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <h1 className="text-3xl font-bold mb-6">Molecule Generator Debug</h1>

      <div className="mb-6">
        <input
          type="text"
          value={inputSmiles}
          onChange={(e) => setInputSmiles(e.target.value)}
          className="bg-gray-800 text-white p-2 rounded mr-2"
          placeholder="Enter SMILES (e.g., CCO)"
        />
        <button onClick={generateMolecule} className="bg-white text-black px-4 py-2 rounded" disabled={loading}>
          {loading ? "Generating..." : "Generate"}
        </button>
      </div>

      {error && (
        <div className="bg-red-900 p-4 rounded mb-6">
          <h2 className="text-xl font-bold mb-2">Error</h2>
          <pre className="whitespace-pre-wrap">{error}</pre>
        </div>
      )}

      {loading && (
        <div className="text-center py-8">
          <MoleculeLoadingAnimation />
        </div>
      )}

      {molecule && !loading && (
        <div className="space-y-8">
          <div className="bg-gray-900 p-6 rounded">
            <h2 className="text-2xl font-bold mb-4">Input Molecule</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p>
                  <span className="text-gray-400">SMILES:</span> {molecule.input_smiles}
                </p>
                <p>
                  <span className="text-gray-400">QED:</span> {molecule.input_qed}
                </p>

                {molecule.input_image && (
                  <div className="mt-4">
                    <h3 className="text-xl mb-2">2D Structure</h3>
                    <div className="bg-white p-2 rounded">
                      <img
                        src={molecule.input_image || "/placeholder.svg"}
                        alt="Input 2D Molecule"
                        className="max-w-full h-auto"
                      />
                    </div>
                  </div>
                )}
              </div>

              <div>
                <h3 className="text-xl mb-2">3D Structure</h3>
                <div className="viewer">
                  <div ref={inputViewerRef} className="w-full h-full"></div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 p-6 rounded">
            <h2 className="text-2xl font-bold mb-4">Generated Molecule</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p>
                  <span className="text-gray-400">SMILES:</span> {molecule.generated_smiles}
                </p>
                <p>
                  <span className="text-gray-400">QED:</span> {molecule.generated_qed}
                </p>
                <p>
                  <span className="text-gray-400">Similarity:</span> {molecule.similarity}
                </p>
                <p>
                  <span className="text-gray-400">LogP:</span> {molecule.LogP}
                </p>
                <p>
                  <span className="text-gray-400">Molecular Weight:</span> {molecule["Molecular Weight"]}
                </p>
                <p>
                  <span className="text-gray-400">TPSA:</span> {molecule.TPSA}
                </p>
                <p>
                  <span className="text-gray-400">H Donors:</span> {molecule.NumHDonors}
                </p>
                <p>
                  <span className="text-gray-400">H Acceptors:</span> {molecule.NumHAcceptors}
                </p>

                {molecule.generated_image && (
                  <div className="mt-4">
                    <h3 className="text-xl mb-2">2D Structure</h3>
                    <div className="bg-white p-2 rounded">
                      <img
                        src={molecule.generated_image || "/placeholder.svg"}
                        alt="Generated 2D Molecule"
                        className="max-w-full h-auto"
                      />
                    </div>
                  </div>
                )}
              </div>

              <div>
                <h3 className="text-xl mb-2">3D Structure</h3>
                <div className="viewer">
                  <div ref={generatedViewerRef} className="w-full h-full"></div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 p-6 rounded">
            <h2 className="text-2xl font-bold mb-4">Raw Response Data</h2>
            <pre className="bg-gray-800 p-4 rounded overflow-auto max-h-[400px] text-xs">
              {JSON.stringify(molecule, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

