import React, { useState, useEffect } from "react";

function App() {
  const [models, setModels] = useState([]); // State to hold deployed models
  const [selectedModel, setSelectedModel] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  // Fetch deployed models when the component mounts
  useEffect(() => {
    fetchDeployedModels();
  }, []);

  const fetchDeployedModels = async () => {
    try {
      const response = await fetch("http://localhost:5000/metrics/");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setModels(Object.keys(data)); // Assuming the response contains a dictionary of metrics keyed by model names
    } catch (error) {
      console.error("Failed to fetch deployed models:", error);
    }
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleInference = async () => {
    if (!selectedModel || !file) {
      alert("Please select a model and upload a file.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("model_name", selectedModel);
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/inference/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      await fetchResults();
    } catch (error) {
      console.error("Inference request failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch("http://localhost:5000/results/");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      console.error("Failed to fetch results:", error);
    }
  };

  const handleFeedback = async (isCorrect) => {
    try {
      const response = await fetch("http://localhost:5000/feedback/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_name: selectedModel, correct: isCorrect }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error("Failed to submit feedback:", error);
    }
  };

  return (
    <div>
      <ModelSelector models={models} setSelectedModel={setSelectedModel} />
      <input type="file" onChange={handleFileChange} />
      <Button onClick={handleInference}>Send Client Files</Button>
      {loading && <p>Loading...</p>}
      {result && (
        <div>
          <h2>Inference Result</h2>
          <p>{result}</p>
          <Button onClick={() => handleFeedback(true)}>Satisfied</Button>
          <Button onClick={() => handleFeedback(false)}>Unsatisfied</Button>
        </div>
      )}
    </div>
  );
}

function ModelSelector({ models, setSelectedModel }) {
  const handleChange = (event) => {
    setSelectedModel(event.target.value);
  };

  return (
    <div>
      <p>Choose a model:</p>
      <select onChange={handleChange}>
        <option value="">Select a model</option>
        {models.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
    </div>
  );
}

function Button({ onClick, children }) {
  return <button onClick={onClick}>{children}</button>;
}

export default App;
