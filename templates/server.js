// i want to add the  following code in my html file
const express = require('express');
const { exec } = require('child_process');
const app = express();
const port = 3000;

app.use(express.static('public')); // Serve static files from 'public' directory

app.get('/run-script', (req, res) => {
  exec('python implement_cvae.py', (error, stdout, stderr) => {
    if (error) {
      console.error(`exec error: ${error}`);
      return res.send(`Error running script: ${error}`);
    }
    res.send(`Script output: ${stdout}`);
  });
});

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`);
});
