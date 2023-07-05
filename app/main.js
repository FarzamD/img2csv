const { app, BrowserWindow } = require('electron');
let mainWidnow = null;
// ignore certificate error
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
driver = webdriver.Chrome(chrome_options=options)

app.on('ready', () => {
	console.log('Hello from Electron');
	mainWidnow = new BrowserWindow();
	mainWidnow.webContents.loadURL(`file://${__dirname}/index.html`);
});