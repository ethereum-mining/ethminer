 contract JSON_Test {
    event Log0(uint value) ;
    event Log0Anonym (uint value) anonymous;
    event Log1(bool indexed aBool, uint value);
    event Log1Anonym(bool indexed aBool, uint value) anonymous;
    event Log2(bool indexed aBool, address indexed aAddress, uint value);
    event Log2Anonym(bool indexed aBool, address indexed aAddress, uint value) anonymous;
    event Log3(bool indexed aBool, address indexed aAddress, bytes32 indexed aBytes32, uint value);
    event Log3Anonym(bool indexed aBool, address indexed aAddress, bytes32 indexed aBytes32, uint value) anonymous;
    event Log4(bool indexed aBool, address indexed aAddress, bytes32 indexed aBytes32, int8 aInt8, uint value);
    event Log4Anonym(bool indexed aBool, address indexed aAddress, bytes32 indexed aBytes32, int8 aInt8, uint value) anonymous;

    function JSON_Test() {
    }

    function setBool(bool _bool) {
        myBool = _bool;
    }
    
    function setInt8(int8 _int8) {
        myInt8 = _int8;
    }
    
    function setUint8(uint8 _uint8) {
        myUint8 = _uint8;
    }
    
    function setInt256(int256 _int256) {
        myInt256 = _int256;
    }
    
    function setUint256(uint256 _uint256) {
        myUint256 = _uint256;
    }
    
    function setAddress(address _address) {
        myAddress = _address;
    }

    function setBytes32(bytes32 _bytes32) {
        myBytes32 = _bytes32;
    }
    
    function getBool() returns (bool ret) {
        return myBool;
    }
    
    function getInt8() returns (int8 ret) {
        return myInt8;
    }
    
    function getUint8() returns (uint8 ret)  {
        return myUint8;
    }
    
    function getInt256() returns (int256 ret) {
        return myInt256;
    }
    
    function getUint256() returns (uint256 ret) {
        return myUint256;
    }
    
    function getAddress() returns (address ret) {
        return myAddress;
    }
    
    function getBytes32() returns (bytes32 ret) {
        return myBytes32;
    }
    
    function fireEventLog0() {
        Log0(42);
    }
    
    function fireEventLog0Anonym() {
        Log0Anonym(42);
    }
    
    function fireEventLog1() {
        Log1(true, 42);
    }
    
    function fireEventLog1Anonym() {
        Log1Anonym(true, 42);
    }
    
    function fireEventLog2() {
        Log2(true, msg.sender, 42);
    }
    
    function fireEventLog2Anonym() {
        Log2Anonym(true, msg.sender, 42);
    }
    
    function fireEventLog3() {
        Log3(true, msg.sender, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff, 42);
    }
    
    function fireEventLog3Anonym() {
        Log3Anonym(true, msg.sender, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff, 42);
    }
    
    function fireEventLog4() {
        Log4(true, msg.sender, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff, -23, 42);
    }
    
    function fireEventLog4Anonym() {
        Log4Anonym(true, msg.sender, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff, -23, 42);
    }

    bool myBool;
    int8 myInt8;
    uint8 myUint8;
    int256 myInt256;
    uint256 myUint256;
    address myAddress;
    bytes32 myBytes32;    
}

