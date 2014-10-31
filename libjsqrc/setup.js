navigator.qt = _web3;

(function () {
	navigator.qt.handlers = [];
	Object.defineProperty(navigator.qt, 'onmessage', {
		set: function (handler) {
			navigator.qt.handlers.push(handler);
		}
	});
})();

navigator.qt.response.connect(function (res) {
	navigator.qt.handlers.forEach(function (handler) {
		handler({data: res});
	});
});

if (window.Promise === undefined) {
	window.Promise = ES6Promise.Promise;
}

web3.setProvider(new web3.providers.QtProvider());

