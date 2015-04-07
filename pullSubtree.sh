# usage 
# ./pullsubtree [repository branch] [repository2 branch2]
#
# example
# ./pullSubtree evmjit master
# ./pullSubtree ethereumjs develop
# ./pullSubtree evmjit master ethereumjs master

evmjit_repo="https://github.com/ethereum/evmjit"
evmjit_location="evmjit"

ethereumjs_repo="https://github.com/ethereum/ethereum.js"
ethereumjs_location="libjsqrc/ethereumjs"

natspecjs_repo="https://github.com/ethereum/natspec.js"
natspecjs_location="libnatspec/natspecjs"

while [ "$1" != "" ]; do
    case $1 in
        evmjit | ethereumjs | natspecjs ) 
            REPO="${1}_repo"
            REPO=${!REPO}
            LOCATION="${1}_location"
            LOCATION=${!LOCATION}
            shift
            BRANCH=$1
            git subtree pull --prefix=${LOCATION} ${REPO} ${BRANCH} --squash
            ;;
    esac
    shift
done

