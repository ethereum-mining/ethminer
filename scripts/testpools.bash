#!/usr/bin/env bash
## vim:set ft=bash ts=4 sw=4 et:
#
# Testscript to test ethminer multiple pools/hosts/syntaxes
# Put this script in the bin directory of ethminer and start running
#
# Run each host 30 times (having a sleep time of 5 sec) which
# means we run one host max 150sec and wait for one of the following
# statements in the output:
#    * Accepted   ==> Poolconnection works
#    * Error      ==> Poolconnection fails
#    * Terminated ==> Poolconnection fails
# If we don't get any of them within the runtime connection is unconfirmed

# As Andrea Lanfranchi wrote a lot of the current startum protocol
# implementation and  pool handling parts we can honor by using his
# donation wallet adresses

# export some vars as "./ethminer" could be still a wrapper script
export ETH_WALLET="0x9E431042fAA3224837e9BEDEcc5F4858cf0390B9"
export WORKERNAME="pooltester"
export EMAIL="andrea.lanfranchi@gmail.com"
export USERNAME="aminer"
export WORKERPWD="x"
export BTC_WALLET="3C4FURwL4oAaEUuCLYmNPUEKQSPR1FAJ3m"


POOLS=""
#2miners.com
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth.2miners.com:2020"
#dwarfpool.org
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-ar.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-asia.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-au.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-br.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-cn.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-cn2.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-eu.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-hk.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-sg.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-ru.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-ru2.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-us.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-us2.dwarfpool.com:8008/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-ar.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-asia.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-au.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-br.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-cn.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-cn2.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-eu.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-hk.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-sg.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-ru.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-ru2.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-us.dwarfpool.com:8008"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-us2.dwarfpool.com:8008"
#ethermine.org
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@asia1.ethermine.org:4444"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eu1.ethermine.org:4444"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@us1.ethermine.org:4444"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@us2.ethermine.org:4444"
#ethermine.org-ssl
POOLS="$POOLS stratum+ssl://ETH_WALLET.WORKERNAME@asia1.ethermine.org:5555"
POOLS="$POOLS stratum+ssl://ETH_WALLET.WORKERNAME@eu1.ethermine.org:5555"
POOLS="$POOLS stratum+ssl://ETH_WALLET.WORKERNAME@us1.ethermine.org:5555"
POOLS="$POOLS stratum+ssl://ETH_WALLET.WORKERNAME@us2.ethermine.org:5555"
#ethpool.org
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@asia1.ethpool.org:3333"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eu1.ethpool.org:3333"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@us1.ethpool.org:3333"
#f2pool.com
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth.f2pool.com:8008"
#miningpoolhub.com
POOLS="$POOLS stratum+tcp://USERNAME.WORKERNAME:WORKERPWD@asia.ethash-hub.miningpoolhub.com:20535"
POOLS="$POOLS stratum+tcp://USERNAME.WORKERNAME:WORKERPWD@europe.ethash-hub.miningpoolhub.com:20535"
POOLS="$POOLS stratum+tcp://USERNAME.WORKERNAME:WORKERPWD@us-east.ethash-hub.miningpoolhub.com:20535"
#nanopool.org
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-asia1.nanopool.org:9999/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-eu1.nanopool.org:9999/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-eu2.nanopool.org:9999/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-us-east1.nanopool.org:9999/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET@eth-us-west1.nanopool.org:9999/WORKERNAME/EMAIL"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-asia1.nanopool.org:9999"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-eu1.nanopool.org:9999"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-eu2.nanopool.org:9999"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-us-east1.nanopool.org:9999"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth-us-west1.nanopool.org:9999"
#nicehash.com
POOLS="$POOLS stratum+tcp://BTC_WALLET@daggerhashimoto.br.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET@daggerhashimoto.eu.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET@daggerhashimoto.hk.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET@daggerhashimoto.in.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET@daggerhashimoto.jp.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET@daggerhashimoto.usa.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.br.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.eu.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.hk.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.in.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.jp.nicehash.com:3353"
POOLS="$POOLS stratum+tcp://BTC_WALLET.WORKERNAME@daggerhashimoto.usa.nicehash.com:3353"
#pool.sexy
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eth.pool.sexy:10002"
#sparkpool.com
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@cn.sparkpool.com:3333"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@eu.sparkpool.com:3333"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@jp.sparkpool.com:3333"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@kr.sparkpool.com:3333"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@na-east.sparkpool.com:3333"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@na-west.sparkpool.com:3333"
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@tw.sparkpool.com:3333"
# whalesburg
POOLS="$POOLS stratum+tcp://ETH_WALLET.WORKERNAME@proxy.pool.whalesburg.com:8082"

# check if any parameter and give a hint to specify -G, -U or -X
if [[ "x" == "x$1" ]]; then
    self=$(basename $0)
    echo "One of -G, -U or -X must be specified"
    exit 2
fi

error_cnt=0
for pool in $POOLS; do
    rm -f log.txt
    current_test_pattern=$pool

    # replace placeholders in the pattern with our values
    pool=$(echo "${pool/ETH_WALLET/$ETH_WALLET}")
    pool=$(echo "${pool/WORKERNAME/$WORKERNAME}")
    pool=$(echo "${pool/EMAIL/$EMAIL}")
    pool=$(echo "${pool/USERNAME/$USERNAME}")
    pool=$(echo "${pool/WORKERPWD/$WORKERPWD}")
    pool=$(echo "${pool/BTC_WALLET/$BTC_WALLET}")

    echo "Testing=$current_test_pattern"
    echo "./ethminer -v 9 --exit -P $pool $@"
    ./ethminer -v 9 --exit -P $pool $@ > log.txt 2>&1 &
    pid=$!
    #echo "PID=$pid"

    exit_due_log=0
    for ((i=0; i<30; i++)) do
        sleep 5                     # 30 * 5sec = 150sec max total running time per host

        l=$(grep "Accepted" log.txt | wc -l)
        if [[ $l != 0 ]]; then
            echo "OK: $current_test_pattern"
            exit_due_log=1
            break
        fi

        l=$(grep "Error" log.txt | wc -l)
        if [[ $l != 0 ]]; then
            cp -a log.txt error${error_cnt}.txt
            error_cnt=$((error_cnt+1))
            echo "ERROR (Error): $current_test_pattern"
            exit_due_log=1
            break
        fi

        l=$(grep "Terminated" log.txt | wc -l)
        if [[ $l != 0 ]]; then
            cp -a log.txt error${error_cnt}.txt
            error_cnt=$((error_cnt+1))
            echo "ERROR (Terminated): $current_test_pattern"
            exit_due_log=1
            break
        fi
    done

    kill -2 $pid
    wait $pid

    if [[ $exit_due_log != 1 ]]; then # seems we've not submitted any share within our mining time - treat as error
        cp -a log.txt error${error_cnt}.txt
        error_cnt=$((error_cnt+1))
        echo "WARNING - UNCONFIRMED STATUS: No share submitted while running: $current_test_pattern"
        echo "                              Fix this by increase runtime or hashrate!"
    fi

    sleep 1
done

if [[ $error_cnt == 0 ]]; then
    echo "SUCCESS: All tests done!"
else
    echo "ERROR: $error_cnt test(s) failed!"
    exit 1
fi

exit 0

