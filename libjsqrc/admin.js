web3.admin = {};
web3.admin.setSessionKey = function(s) { web3.admin.sessionKey = s; };

var getSessionKey = function () { return web3.admin.sessionKey; };

web3._extend({
    property: 'admin',
    methods: [new web3._extend.Method({
        name: 'web3.setVerbosity',
        call: 'admin_web3_setVerbosity',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'net.start',
        call: 'admin_net_start',
        inputFormatter: [getSessionKey],
        params: 1
    }), new web3._extend.Method({
        name: 'net.stop',
        call: 'admin_net_stop',
        inputFormatter: [getSessionKey],
        params: 1
    }), new web3._extend.Method({
        name: 'net.connect',
        call: 'admin_net_connect',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'net.peers',
        call: 'admin_net_peers',
        inputFormatter: [getSessionKey],
        params: 1
    }), new web3._extend.Method({
        name: 'eth.blockQueueStatus',
        call: 'admin_eth_blockQueueStatus',
        inputFormatter: [getSessionKey],
        params: 1
    }), new web3._extend.Method({
        name: 'net.nodeInfo',
        call: 'admin_net_nodeInfo',
        inputFormatter: [getSessionKey],
        params: 1
    }), new web3._extend.Method({
        name: 'eth.setAskPrice',
        call: 'admin_eth_setAskPrice',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.setBidPrice',
        call: 'admin_eth_setBidPrice',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.setReferencePrice',
        call: 'admin_eth_setReferencePrice',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.setPriority',
        call: 'admin_eth_setPriority',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.setMining',
        call: 'admin_eth_setMining',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.findBlock',
        call: 'admin_eth_findBlock',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.blockQueueFirstUnknown',
        call: 'admin_eth_blockQueueFirstUnknown',
        inputFormatter: [getSessionKey],
        params: 1
    }), new web3._extend.Method({
        name: 'eth.blockQueueRetryUnknown',
        call: 'admin_eth_blockQueueRetryUnknown',
        inputFormatter: [getSessionKey],
        params: 1
    }), new web3._extend.Method({
        name: 'eth.allAccounts',
        call: 'admin_eth_allAccounts',
        inputFormatter: [getSessionKey],
        params: 1
    }), new web3._extend.Method({
        name: 'eth.newAccount',
        call: 'admin_eth_newAccount',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.setSigningKey',
        call: 'admin_eth_setSigningKey',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.setMiningBenefactor',
        call: 'admin_eth_setMiningBenefactor',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.inspect',
        call: 'admin_eth_inspect',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.reprocess',
        call: 'admin_eth_reprocess',
        inputFormatter: [null, getSessionKey],
        params: 2
    }), new web3._extend.Method({
        name: 'eth.vmTrace',
        call: 'admin_eth_vmTrace',
        inputFormatter: [null, null, getSessionKey],
        params: 3
    }), new web3._extend.Method({
        name: 'eth.getReceiptByHashAndIndex',
        call: 'admin_eth_getReceiptByHashAndIndex',
        inputFormatter: [null, null, getSessionKey],
        params: 3
    })]
});

