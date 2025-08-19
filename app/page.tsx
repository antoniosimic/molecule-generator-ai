"use client"

// Add this at the top of the file or in a global declaration file
declare global {
  interface Window {
    $3Dmol?: any; // Koristimo any jer nemamo specifične TypeScript definicije za $3Dmol
  }
}

import { useState, useEffect, useRef, MutableRefObject } from "react"
import axios from "axios"
import { Loader2, BrainCircuit, FlaskConical } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { motion } from "framer-motion"
import Head from "next/head"
import MoleculeLoadingAnimation from "@/components/molecule-loading-animation"

interface MoleculeData {
  input_smiles: string
  input_qed: number
  input_image: string
  input_mol_block: string
  generated_smiles: string
  generated_qed: number
  similarity: number
  generated_image: string
  generated_mol_block: string
  LogP: number
  "Molecular Weight": number
  TPSA: number
  NumHDonors: number
  NumHAcceptors: number
  model_used?: string
  model_log?: string
}

const MODEL_IDS = ["model1", "model2", "model3", "model4"]

export default function Home() {
  const [inputSmiles, setInputSmiles] = useState("CCO")
  const [molecule, setMolecule] = useState<MoleculeData | null>(null)
  const [loading, setLoading] = useState(false)
  const [is3DmolLoaded, setIs3DmolLoaded] = useState(false)
  const [selectedModelId, setSelectedModelId] = useState<string>(MODEL_IDS[0])

  const inputViewerRef = useRef<HTMLDivElement | null>(null)
  const generatedViewerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (window.$3Dmol) {
      console.log("3Dmol.js is already available.");
      setIs3DmolLoaded(true);
      return;
    }

    const script = document.createElement("script");
    script.src = "https://3Dmol.org/build/3Dmol-min.js";
    script.async = true;
    script.onload = () => {
      console.log("3Dmol.js loaded successfully via script tag.");
      setIs3DmolLoaded(true);
    };
    script.onerror = () => {
      console.error("Failed to load 3Dmol.js. Check network or script URL.");
      setIs3DmolLoaded(false);
    };
    document.head.appendChild(script);
  }, []);

  const generateMolecule = async () => {
    if (!inputSmiles.trim()) {
        alert("Please enter a SMILES string.");
        return;
    }
    if (!selectedModelId) {
        alert("Please select a model.");
        return;
    }

    setLoading(true);
    setMolecule(null);
    console.log(`Requesting molecule generation for SMILES: ${inputSmiles} with Model: ${selectedModelId}`);
    try {
      const response = await axios.post("/api/generate", {
        smiles: inputSmiles,
        model_id: selectedModelId,
      });
      console.log("Received data from backend:", response.data);
      setMolecule(response.data);
    } catch (error: any) {
      console.error("Error generating molecule:", error);
      const msg = error.response?.data?.error || "An error occurred while generating the molecule.";
      const det = error.response?.data?.details ? `\nDetails: ${error.response.data.details}` : "";
      alert(`${msg}${det}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!molecule) {
      return;
    }
    if (!is3DmolLoaded) {
      console.log("Viewer setup skipped: 3Dmol.js not loaded yet.");
      return;
    }
    if (!window.$3Dmol) {
      console.warn("Viewer setup skipped: 3Dmol.js reported as loaded, but window.$3Dmol is not available.");
      return;
    }

    console.log("Attempting to set up 3D viewers.");

    const setupViewer = (
      viewerRef: MutableRefObject<HTMLDivElement | null>,
      molBlock: string | undefined | null,
      title: string
    ) => {
      const el = viewerRef.current;
      if (!el) {
        console.error(`Target DOM element for ${title} viewer not found.`);
        return;
      }
      el.innerHTML = ""; 

      if (molBlock && molBlock.trim() !== "") {
        console.log(`Setting up viewer for ${title} with molBlock data.`);
        try {
          const viewer = window.$3Dmol.createViewer(el, { backgroundColor: "0xffffff" });
          viewer.addModel(molBlock, "mol");
          viewer.setStyle({}, {
            stick: { radius: 0.15, colorscheme: "jmol" },
            sphere: { radius: 0.5, colorscheme: "jmol" },
          });
          viewer.zoomTo();
          viewer.render();
          console.log(`${title} viewer created successfully.`);
        } catch (e) {
          console.error(`Error creating 3Dmol viewer for ${title}:`, e);
          el.innerHTML = `<div class="flex items-center justify-center h-full text-red-500 p-4 text-center">Error rendering 3D view for ${title}.</div>`;
        }
      } else {
        console.log(`No valid 3D data for ${title}. Displaying fallback message.`);
        el.innerHTML = `<div class="flex items-center justify-center h-full text-gray-500 p-4 text-center">No 3D data available for ${title}.</div>`;
      }
    };

    if (inputViewerRef.current) {
        setupViewer(inputViewerRef, molecule.input_mol_block, "Input Molecule");
    }
    if (generatedViewerRef.current) {
        setupViewer(generatedViewerRef, molecule.generated_mol_block, "Generated Molecule");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [molecule, is3DmolLoaded]);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 100 },
    },
  };

  return (
    <>
    <Head>
        <title>Molecule AI Generator</title>
        <meta name="description" content="Generate novel molecular structures using AI and Reinforcement Learning." />
    </Head>
    <div className="min-h-screen bg-black text-white flex flex-col items-center p-4 sm:p-8">
      <motion.div
        className="w-full max-w-6xl mx-auto"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <motion.div
          className="text-center mb-10 sm:mb-12"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h1 className="text-4xl sm:text-5xl font-bold mb-3 sm:mb-4 bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 bg-clip-text text-transparent">
            Molecule Generator AI
          </h1>
          <p className="text-gray-400 text-lg sm:text-xl">
            Discover and visualize molecular structures with precision using AI.
          </p>
        </motion.div>

        <motion.div
          className="bg-gray-900 bg-opacity-70 backdrop-blur-md p-6 sm:p-8 rounded-xl shadow-2xl space-y-6 mb-10 sm:mb-12 max-w-2xl mx-auto"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={itemVariants} className="space-y-2">
            <label htmlFor="smilesInput" className="text-sm font-medium text-gray-300 flex items-center">
              <FlaskConical size={18} className="mr-2 text-purple-400" />
              Enter SMILES String
            </label>
            <Input
              id="smilesInput"
              type="text"
              placeholder="e.g., CCO for Ethanol"
              value={inputSmiles}
              onChange={(e) => setInputSmiles(e.target.value)}
              className="bg-gray-800 border-gray-700 text-white placeholder-gray-500 focus:ring-purple-500 focus:border-purple-500"
            />
          </motion.div>

          <motion.div variants={itemVariants} className="space-y-2">
            <label className="text-sm font-medium text-gray-300 flex items-center">
              <BrainCircuit size={18} className="mr-2 text-purple-400" />
              Select Model
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {MODEL_IDS.map((id) => (
                <Button
                  key={id}
                  onClick={() => setSelectedModelId(id)}
                  variant={selectedModelId === id ? "default" : "outline"}
                  className={`w-full transition-colors duration-200 ${
                    selectedModelId === id
                      ? "bg-purple-600 hover:bg-purple-700 text-white border-purple-600"
                      : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700 hover:border-gray-500"
                  }`}
                >
                  {`M ${id.slice(-1)}`}
                </Button>
              ))}
            </div>
          </motion.div>

          <motion.div variants={itemVariants}>
            <Button
              onClick={generateMolecule}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white py-3 text-lg transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-70"
              disabled={loading || !inputSmiles.trim()}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Generating...
                </>
              ) : (
                `Generate with Model ${selectedModelId.slice(-1)}`
              )}
            </Button>
          </motion.div>
        </motion.div>

        {loading && (
          <motion.div
            className="flex justify-center items-center py-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <MoleculeLoadingAnimation />
          </motion.div>
        )}

        {molecule && !loading && (
          <motion.div 
            variants={containerVariants} 
            initial="hidden" 
            animate="visible" 
            className="space-y-8 md:space-y-0 md:grid md:grid-cols-2 md:gap-8"
          >
            <motion.div variants={itemVariants}>
              <Card className="bg-gray-900 border-gray-700 overflow-hidden h-full flex flex-col">
                <CardHeader className="bg-gray-800 border-b border-gray-700">
                  <CardTitle className="text-xl text-purple-300">Input Molecule</CardTitle>
                </CardHeader>
                <CardContent className="pt-6 flex-grow flex flex-col">
                  <div className="space-y-3 mb-4">
                    <div>
                      <p className="text-xs text-gray-400 mb-1">SMILES</p>
                      <p className="font-mono text-sm bg-gray-800 p-2 rounded overflow-x-auto break-all">
                        {molecule.input_smiles}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400 mb-1">QED</p>
                      <p className="font-semibold">
                        {molecule.input_qed !== undefined ? molecule.input_qed.toFixed(4) : "N/A"}
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-4 mt-auto">
                    {molecule.input_image && (
                      <div className="mb-2">
                        <p className="text-xs text-gray-400 mb-1">2D Structure</p>
                        <div className="bg-white p-2 rounded aspect-square max-w-[200px] mx-auto flex items-center justify-center">
                          <img
                            src={molecule.input_image}
                            alt="Input 2D Molecule"
                            className="max-w-full max-h-full object-contain"
                          />
                        </div>
                      </div>
                    )}
                     <div className="space-y-1">
                        <p className="text-xs text-gray-400 mb-1">3D Structure</p>
                        <div 
                            className="w-full h-64 sm:h-72 bg-white rounded border border-gray-600"
                            style={{ minHeight: '250px' }}
                        >
                          {/* KLJUČNA PROMJENA OVDJE: dodana 'relative' klasa */}
                          <div ref={inputViewerRef} className="w-full h-full relative"></div>
                        </div>
                      </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div variants={itemVariants}>
              <Card className="bg-gray-900 border-gray-700 overflow-hidden h-full flex flex-col">
                <CardHeader className="bg-gray-800 border-b border-gray-700">
                  <CardTitle className="text-xl text-green-400">
                    Generated Molecule (Model {molecule.model_used?.slice(-1) || selectedModelId.slice(-1)})
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-6 flex-grow flex flex-col">
                  <div className="space-y-3 mb-4">
                    <div>
                      <p className="text-xs text-gray-400 mb-1">SMILES</p>
                      <p className="font-mono text-sm bg-gray-800 p-2 rounded overflow-x-auto break-all">
                        {molecule.generated_smiles}
                      </p>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                      <div>
                        <p className="text-xs text-gray-400">QED</p>
                        <p className="font-semibold">
                          {molecule.generated_qed !== undefined ? molecule.generated_qed.toFixed(4) : "N/A"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Similarity</p>
                        <p className="font-semibold">
                          {molecule.similarity !== undefined ? molecule.similarity.toFixed(4) : "N/A"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">LogP</p>
                        <p className="font-semibold">{molecule.LogP !== undefined ? molecule.LogP.toFixed(2) : "N/A"}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Mol. Weight</p>
                        <p className="font-semibold">
                          {molecule["Molecular Weight"] !== undefined
                            ? molecule["Molecular Weight"].toFixed(2)
                            : "N/A"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">TPSA</p>
                        <p className="font-semibold">{molecule.TPSA !== undefined ? molecule.TPSA.toFixed(2) : "N/A"}</p>
                      </div>
                       <div>
                        <p className="text-xs text-gray-400">H Donors</p>
                        <p className="font-semibold">{molecule.NumHDonors !== undefined ? molecule.NumHDonors : "N/A"}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">H Acceptors</p>
                        <p className="font-semibold">
                          {molecule.NumHAcceptors !== undefined ? molecule.NumHAcceptors : "N/A"}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 gap-4 mt-auto">
                    {molecule.generated_image && (
                      <div className="mb-2">
                        <p className="text-xs text-gray-400 mb-1">2D Structure</p>
                        <div className="bg-white p-2 rounded aspect-square max-w-[200px] mx-auto flex items-center justify-center">
                          <img
                            src={molecule.generated_image}
                            alt="Generated 2D Molecule"
                            className="max-w-full max-h-full object-contain"
                          />
                        </div>
                      </div>
                    )}
                    <div className="space-y-1">
                        <p className="text-xs text-gray-400 mb-1">3D Structure</p>
                        <div 
                            className="w-full h-64 sm:h-72 bg-white rounded border border-gray-600"
                            style={{ minHeight: '250px' }}
                        >
                          {/* KLJUČNA PROMJENA OVDJE: dodana 'relative' klasa */}
                          <div ref={generatedViewerRef} className="w-full h-full relative"></div>
                        </div>
                    </div>
                  </div>
                  {molecule.model_log && (
                    <div className="mt-4">
                      <p className="text-xs text-gray-400 mb-1">Model Log</p>
                      <pre className="text-xs bg-gray-800 p-3 rounded-md text-gray-300 max-h-24 overflow-y-auto whitespace-pre-wrap break-all">
                        {molecule.model_log}
                      </pre>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        )}
      </motion.div>
    </div>
    </>
  )
}