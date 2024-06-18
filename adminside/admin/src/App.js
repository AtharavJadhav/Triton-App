import React, { useState, useEffect } from "react";

function App() {
  const [selectedModels, setSelectedModels] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [selectedModelForShift, setSelectedModelForShift] = useState("");

  const handleDeploy = async () => {
    try {
      const response = await fetch("http://localhost:5000/deploy/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ models: selectedModels }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error(
          `HTTP error! status: ${response.status}`,
          errorData.detail
        );
      } else {
        const data = await response.json();
        console.log(data.message);
      }
    } catch (error) {
      console.error("There was a problem with the fetch operation:", error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch("http://localhost:5000/metrics/");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error("Failed to fetch metrics:", error);
    }
  };

  const handleCalculateDataShift = async () => {
    if (!selectedModelForShift) {
      console.error("No model selected for data shift calculation");
      return;
    }

    try {
      const response = await fetch(
        "http://localhost:5000/calculate_shift_metrics/",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ model_name: selectedModelForShift }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error(
          `HTTP error! status: ${response.status}`,
          errorData.detail
        );
      } else {
        const data = await response.json();
        console.log("Data shift metrics:", data);
        // Optionally update the state with the new data shift metrics
        setMetrics((prevMetrics) => ({
          ...prevMetrics,
          [data.model_name]: {
            ...prevMetrics[data.model_name],
            data_shift: data.data_shift,
            mean_shift: data.mean_shift,
            std_shift: data.std_shift,
          },
        }));
      }
    } catch (error) {
      console.error("Failed to calculate data shift:", error);
    }
  };

  useEffect(() => {
    const intervalId = setInterval(fetchMetrics, 5000); // Fetch metrics every 5 seconds
    return () => clearInterval(intervalId); // Clean up interval on unmount
  }, []);

  return (
    <div>
      <ModelDeploy
        setSelectedModels={setSelectedModels}
        handleDeploy={handleDeploy}
      />
      <ModelMetrics
        metrics={metrics}
        setSelectedModelForShift={setSelectedModelForShift}
        handleCalculateDataShift={handleCalculateDataShift}
      />
    </div>
  );
}

function ModelDeploy({ setSelectedModels, handleDeploy }) {
  const handleCheckboxChange = (event) => {
    const value = event.target.value;
    setSelectedModels((prev) => {
      if (event.target.checked) {
        return [...prev, value];
      } else {
        return prev.filter((model) => model !== value);
      }
    });
  };

  return (
    <div>
      <p>Select models to deploy:</p>
      <input
        type="checkbox"
        id="mnist_model_onnx"
        name="mnist_model_onnx"
        value="mnist_model_onnx"
        onChange={handleCheckboxChange}
      />
      <label htmlFor="mnist_model_onnx">MNIST ONNX</label>
      <br />
      <input
        type="checkbox"
        id="mnist_model_openvino"
        name="mnist_model_openvino"
        value="mnist_model_openvino"
        onChange={handleCheckboxChange}
      />
      <label htmlFor="mnist_model_openvino">MNIST OpenVino</label>
      <br />
      <input
        type="checkbox"
        id="mnist_model_pytorch"
        name="mnist_model_pytorch"
        value="mnist_model_pytorch"
        onChange={handleCheckboxChange}
      />
      <label htmlFor="mnist_model_pytorch">MNIST Pytorch</label>
      <br />
      <input
        type="checkbox"
        id="bert_model_onnx"
        name="bert_model_onnx"
        value="bert_model_onnx"
        onChange={handleCheckboxChange}
      />
      <label htmlFor="bert_model_onnx">BERT ONNX</label>
      <br />
      <Button onClick={handleDeploy}>Deploy</Button>
    </div>
  );
}

function ModelMetrics({
  metrics,
  setSelectedModelForShift,
  handleCalculateDataShift,
}) {
  return (
    <div>
      <h2>Model Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>Model Name</th>
            <th>Accuracy</th>
            <th>Latency (s)</th>
            <th>Throughput (req/s)</th>
            <th>Request Count</th>
            <th>Data Shift</th>
            <th>Mean Shift</th>
            <th>Std Shift</th>
            <th>Calculate Data Shift</th>
          </tr>
        </thead>
        <tbody>
          {Object.keys(metrics).map((modelName) => (
            <tr key={modelName}>
              <td>{modelName}</td>
              <td>{(metrics[modelName].accuracy * 100).toFixed(2)}%</td>
              <td>{metrics[modelName].latency.toFixed(4)}</td>
              <td>{metrics[modelName].throughput.toFixed(2)}</td>
              <td>{metrics[modelName].request_count}</td>
              <td>{metrics[modelName].data_shift.toFixed(4)}</td>
              <td>{metrics[modelName].mean_shift.toFixed(4)}</td>
              <td>{metrics[modelName].std_shift.toFixed(4)}</td>
              <td>
                <button
                  onClick={() => {
                    setSelectedModelForShift(modelName);
                    handleCalculateDataShift();
                  }}
                >
                  Calculate
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Button({ onClick, children }) {
  return <button onClick={onClick}>{children}</button>;
}

export default App;
