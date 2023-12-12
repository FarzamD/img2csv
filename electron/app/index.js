// const fs= require('fs'); //error require not defined
// import { fs } from "fs"; //Cannot use import statement outside a module
function loadImg(e){
    var File = e.target.files[0];
    if (File) {
      // console.log('File:')
      // console.log(File)
      var img = document.getElementById('img');
      img.title= File.name;
      img.src= File.path;
      img.width= 300;      
    }
}
function readJSON(path) {
  return fetch(path)
    .then(response => response.json())
    .then(data => {
      // Assign the parsed data to the global variable
      lines = data;
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

function detectLines() {
    var canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    console.log('det lines');
}
let lines; // Declare a global variable

// Usage example
readJSON('../lines.json')
  .then(() => {
    console.log('Global variable now contains JSON data:', lines);
  });
var l=readJSON('../lines.json')
  .then(() => {
    return lines;
});

console.log(lines);
console.log(l);
console.data