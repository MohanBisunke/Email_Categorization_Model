const express = require("express");
const { PythonShell } = require("python-shell");
const app = express();
const port = 5000;

app.use(express.json());
app.post("/predict", async (req, res) => {

  try {
    const outputMessage = await PythonShell.run("./script.py",{
        args: ["--subject", req.body.subject, "--message", req.body.message]
    });
    // console.log(outputMessage[outputMessage.length - 1]);
    res.status(200).json({
        predictions: outputMessage[outputMessage.length - 3].split(":")[1],
        predictedClass: outputMessage[outputMessage.length - 2].split(":")[1],
        predictionProbability: outputMessage[outputMessage.length - 1].split(":")[1],
    });

    // const folderPath = './uploads';
    // removeFiles(folderPath);
  } catch (error) {
    console.log("Error:", error);
    res.status(500).send("Internal server error.");
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});