const fs = require('fs');
const path = require('path');
const download = require('download');

const downloadSource = require('./source.json');
const cwd = process.cwd();

(function startDownload(downloadSource) {
  Promise.all(
    downloadSource.files.map(file => {
      let filePath = path.join(cwd, file.destination, file.filename);

      if (!fs.existsSync(filePath)) {
        console.log('downloading: ', file.filename);
        return download(file.url, file.destination, file);
      }
    })
  ).then(() => {
    console.log('All files downloaded!');
  });
})(downloadSource);
