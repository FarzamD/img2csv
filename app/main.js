const { app, BrowserWindow } = require('electron');
const url = require('url');
const path = require('path');

let mainWidnow = null;
// ignore certificate error

app.on('ready', () => {
	app.commandLine.appendSwitch('ignore-certificate-errors');

	console.log('Hello from Electron');
	mainWidnow = new BrowserWindow({
				height: 800,
				width: 600,
				show: false,
				frame: false,
				backgroundColor: '#8c8a7a;'});

	mainWidnow.webContents.loadURL(
					url.format({
						pathname: path.join(__dirname, 'index.html'),
						protocol: 'file:',
						slashes: true ///
					})
				);
			
	mainWidnow.setBackgroundColor('#8c8a7a');

	mainWidnow.once('ready-to-show', () => {
		mainWidnow.show();
	});
	mainWidnow.on('closed', () => {
		mainWidnow = null;
	});
	









});

