const { app, BrowserWindow } = require('electron');
// const { pythonShell } = require('python-shell');
const url = require('url');
const path = require('path');

const createWindow = () => {
	const mainWidnow = new BrowserWindow({
		height: 600,
		width: 450,
		minHeight: 500,
		minWidth: 400,
		maxHeight: 700,
		maxWidth: 500,
		show: false,
		title: 'Img 2 CSV',
		// titleBarStyle: 'hiddenInset',
		backgroundColor: '#8c8a7a;'})
	mainWidnow.loadFile(path.join(__dirname, 'index.html'))
	mainWidnow.once('ready-to-show', () => {
		mainWidnow.show();
	});

}
app.on('ready', () => {
	app.commandLine.appendSwitch('ignore-certificate-errors');

	console.log('Hello from Electron');

	createWindow()

	app.on('activate', () => {
		if (BrowserWindow.getAllWindows().length === 0) {
		  createWindow()
		}
	  })

	// mainWidnow.setBackgroundColor('#8c8a7a')

	// mainWidnow.on('closed', () => {
	// 	ExitApp();
	// 	mainWidnow = null;
	// });
	// function ExitApp() {
	// 	console.log('Exit');
	// }

});

