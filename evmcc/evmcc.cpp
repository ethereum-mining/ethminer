#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libevmface/Instruction.h>

#include <boost/algorithm/string.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace dev;

class EVMCCompiler
{

private:

    struct
    {
        llvm::Type* word8;
        llvm::Type* word8ptr;
        llvm::Type* word256;
        llvm::Type* word256ptr;
        llvm::Type* word256arr;
        llvm::Type* size;
    } Types;

public:

    EVMCCompiler()
    {
        auto& context = llvm::getGlobalContext();
        Types.word8 = llvm::Type::getInt8Ty(context);
        Types.word8ptr = llvm::Type::getInt8PtrTy(context);
        Types.word256 = llvm::Type::getIntNTy(context, 256);
        Types.word256ptr = Types.word256->getPointerTo();
        Types.word256arr = llvm::ArrayType::get(Types.word256, 100);
        Types.size = llvm::Type::getInt64Ty(context);
    }

    void compile(const bytes& bytecode)
    {
        using namespace llvm;

        auto& context = getGlobalContext();

        Module* module = new Module("main", context);
        IRBuilder<> builder(context);

        // Create globals for memory, memory size, stack and stack top
        auto memory = new GlobalVariable(*module, Types.word8ptr, false,
                                         GlobalValue::LinkageTypes::PrivateLinkage,
                                         Constant::getNullValue(Types.word8ptr), "memory");
        auto memSize = new GlobalVariable(*module, Types.size, false,
                                          GlobalValue::LinkageTypes::PrivateLinkage,
                                          ConstantInt::get(Types.size, 0), "memsize");
        auto stack = new GlobalVariable(*module, Types.word256arr, false,
                                        GlobalValue::LinkageTypes::PrivateLinkage,
                                        ConstantAggregateZero::get(Types.word256arr), "stack");
        auto stackTop = new GlobalVariable(*module, Types.size, false,
                                           GlobalValue::LinkageTypes::PrivateLinkage,
                                           ConstantInt::get(Types.size, 0), "stackTop");

        // Create value for void* malloc(size_t)
        std::vector<Type*> mallocArgTypes = { Types.size };
        Value* mallocVal = Function::Create(FunctionType::get(Types.word8ptr, mallocArgTypes, false),
                                            GlobalValue::LinkageTypes::ExternalLinkage, "malloc", module);

        // Create main function
        FunctionType* funcType = FunctionType::get(llvm::Type::getInt32Ty(context), false);
        Function* mainFunc = Function::Create(funcType, Function::ExternalLinkage, "main", module);

        BasicBlock* entryBlock = BasicBlock::Create(context, "entry", mainFunc);
        builder.SetInsertPoint(entryBlock);

        // Initialize memory with call to malloc, update memsize
        std::vector<Value*> mallocMemArgs = { ConstantInt::get(Types.size, 100) };
        auto mallocMemCall = builder.CreateCall(mallocVal, mallocMemArgs, "malloc_mem");
        builder.CreateStore(mallocMemCall, memory);
        builder.CreateStore(ConstantInt::get(Types.size, 100), memSize);

        /*
         std::vector<Value*> mallocStackArgs = { ConstantInt::get(sizeTy, 200) };
         auto mallocStackCall = builder.CreateCall(mallocVal, mallocStackArgs, "malloc_stack");
         auto mallocCast = builder.CreatePointerBitCastOrAddrSpaceCast(mallocStackCall, int256ptr);
         builder.CreateStore(mallocCast, stackVal);
	*/

        builder.CreateRet(ConstantInt::get(Type::getInt32Ty(context), 0));

        module->dump();
    }

};

void show_usage()
{
    // FIXME: Use arg[0] as program name?
    std::cerr << "usage: evmcc (-b|-c|-d)+ <inputfile.bc>\n";
}

int main(int argc, char** argv)
{

    std::string input_file;
    bool opt_dissassemble = false;
    bool opt_show_bytes = false;
    bool opt_compile = false;
    bool opt_unknown = false;

    for (int i = 1; i < argc; i++)
    {
        std::string option = argv[i];
        if (option == "-b")
        {
            opt_show_bytes = true;
        }
        else if (option == "-c")
        {
            opt_compile = true;
        }
        else if (option == "-d")
        {
            opt_dissassemble = true;
        }
        else if (option[0] != '-' && input_file.empty())
        {
            input_file = option;
        }
        else
        {
            opt_unknown = true;
            break;
        }
    }

    if (opt_unknown ||
        input_file.empty() ||
        (!opt_show_bytes && !opt_compile && !opt_dissassemble))
    {
        show_usage();
        exit(1);
    }

    std::ifstream ifs(input_file);
    if (!ifs.is_open())
    {
        std::cerr << "cannot open file " << input_file << std::endl;
        exit(1);
    }

    std::string src((std::istreambuf_iterator<char>(ifs)),
		    (std::istreambuf_iterator<char>()));

    boost::algorithm::trim(src);

    bytes bytecode = fromHex(src);

    if (opt_show_bytes)
    {
        std::cout << dev::memDump(bytecode) << std::endl;
    }

    if (opt_dissassemble)
    {
        std::string assembly = eth::disassemble(bytecode);
        std::cout << assembly << std::endl;
    }

    if (opt_compile)
    {
        EVMCCompiler().compile(bytecode);
    }

    return 0;
}
