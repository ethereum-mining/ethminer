import os
os.chdir('..')

from setuptools import setup, Extension

from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

setup(
    # Name of this package
    name="ethereum-solidity",

    # Package version
    version='1.8.0',

    description='Solidity compiler python wrapper',
    maintainer='Vitalik Buterin',
    maintainer_email='v@buterin.com',
    license='WTFPL',
    url='http://www.ethereum.org/',

    # Describes how to build the actual extension module from C source files.
    ext_modules=[
        Extension(
            'solidity',         # Python name of the module
            sources= ['libdevcore/Common.cpp', 'libdevcore/CommonData.cpp', 'libdevcore/CommonIO.cpp', 'libdevcore/FixedHash.cpp', 'libdevcore/Guards.cpp', 'libdevcore/Log.cpp', 'libdevcore/RangeMask.cpp', 'libdevcore/RLP.cpp', 'libdevcore/Worker.cpp', 'libdevcrypto/AES.cpp', 'libdevcrypto/Common.cpp', 'libdevcrypto/CryptoPP.cpp', 'libdevcrypto/ECDHE.cpp', 'libdevcrypto/FileSystem.cpp', 'libdevcrypto/MemoryDB.cpp', 'libdevcrypto/OverlayDB.cpp', 'libdevcrypto/SHA3.cpp', 'libdevcrypto/TrieCommon.cpp', 'libdevcrypto/TrieDB.cpp', 'libethcore/CommonEth.cpp', 'libethcore/CommonJS.cpp', 'libethcore/Exceptions.cpp', 'libsolidity/AST.cpp', 'libsolidity/ASTJsonConverter.cpp', 'libsolidity/ASTPrinter.cpp', 'libsolidity/CompilerContext.cpp', 'libsolidity/Compiler.cpp', 'libsolidity/CompilerStack.cpp', 'libsolidity/CompilerUtils.cpp', 'libsolidity/DeclarationContainer.cpp', 'libsolidity/ExpressionCompiler.cpp', 'libsolidity/GlobalContext.cpp', 'libsolidity/InterfaceHandler.cpp', 'libsolidity/NameAndTypeResolver.cpp', 'libsolidity/Parser.cpp', 'libsolidity/Scanner.cpp', 'libsolidity/SourceReferenceFormatter.cpp', 'libsolidity/Token.cpp', 'libsolidity/Types.cpp', 'libevmcore/Assembly.cpp', 'libevmcore/Instruction.cpp', 'pysol/pysolidity.cpp'],
            libraries=['boost_python', 'boost_filesystem', 'boost_chrono', 'boost_thread', 'cryptopp', 'leveldb', 'jsoncpp'],
            include_dirs=['/usr/include/boost', '..', '../..', '.'],
            extra_compile_args=['--std=c++11', '-Wno-unknown-pragmas']
        )],
    py_modules=[
    ],
    scripts=[
    ],
    entry_points={
    }
    ),
