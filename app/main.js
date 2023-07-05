const { app, BrowserWindow } = require('electron');
let mainWidnow = null;
// ignore certificate error
app.commandLine.appendSwitch('ignore-certificate-errors');

app.on('ready', () => {
	console.log('Hello from Electron');
	mainWidnow = new BrowserWindow();
	mainWidnow.webContents.loadURL(`file://${__dirname}/index.html`);
});