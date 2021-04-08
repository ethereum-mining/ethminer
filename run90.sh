#!/bin/sh
./build/ethminer/ethminer\
        --pool stratum://0xA5420BeAF1d6fA544fC47f1110D194458659C1A2:@us1.ethermine.org:4444\
                --cuda\
                --cu-streams 1 --cu-target-usage 0.90

