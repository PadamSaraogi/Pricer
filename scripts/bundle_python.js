const fs = require('fs');
const path = require('path');

const backendDir = path.join(__dirname, '..', 'backend');
const outputFile = path.join(__dirname, '..', 'src', 'python_files.json');

const files = {};
const entries = fs.readdirSync(backendDir, { withFileTypes: true });

entries.forEach(entry => {
  if (entry.isFile() && entry.name.endsWith('.py')) {
    const content = fs.readFileSync(path.join(backendDir, entry.name), 'utf-8');
    files[entry.name] = content;
  }
});

let requirements = [];
const reqPath = path.join(backendDir, 'requirements.txt');
if (fs.existsSync(reqPath)) {
  requirements = fs.readFileSync(reqPath, 'utf-8')
    .split('\n')
    .map(r => r.trim())
    .filter(r => r && !r.startsWith('#') && !r.startsWith('streamlit'));
}

fs.writeFileSync(outputFile, JSON.stringify({ files, requirements }, null, 2));
console.log(`Bundled ${Object.keys(files).length} Python files into ${outputFile}`);
