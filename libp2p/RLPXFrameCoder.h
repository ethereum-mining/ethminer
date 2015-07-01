/*
 This file is part of cpp-ethereum.
 
 cpp-ethereum is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 cpp-ethereum is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
 */
/** @file RLPXFrameCoder.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */


#pragma once

#include <memory>
#include <libdevcore/Guards.h>
#include <libdevcrypto/ECDHE.h>
#include <libdevcrypto/CryptoPP.h>
#include "Common.h"

namespace dev
{
namespace p2p
{

struct RLPXFrameInfo
{
	RLPXFrameInfo() = default;
	/// Constructor. frame-size || protocol-type, [sequence-id[, total-packet-size]]
	RLPXFrameInfo(bytesConstRef _frameHeader);
	uint32_t length = 0;			///< Max: 2**24
	uint8_t padding = 0;
	
	uint16_t protocolId = 0;
	bool hasSequence = false;
	uint16_t sequenceId = 0;
	uint32_t totalLength = 0;
};

class RLPXHandshake;

/**
 * @brief Encoder/decoder transport for RLPx connection established by RLPXHandshake.
 *
 * Thread Safety 
 * Distinct Objects: Safe.
 * Shared objects: Unsafe.
 */
class RLPXFrameCoder
{
	friend class RLPXFrameIOMux;
	friend class Session;
public:
	/// Constructor.
	/// Requires instance of RLPXHandshake which has completed first two phases of handshake.
	RLPXFrameCoder(RLPXHandshake const& _init);
	~RLPXFrameCoder() {}
	
	/// Encrypt _packet as RLPx frame.
	void writeSingleFramePacket(bytesConstRef _packet, bytes& o_bytes);

	/// Authenticate and decrypt header in-place.
	bool authAndDecryptHeader(bytesRef io_cipherWithMac);
	
	/// Authenticate and decrypt frame in-place.
	bool authAndDecryptFrame(bytesRef io_cipherWithMac);
	
	/// Return first 16 bytes of current digest from egress mac.
	h128 egressDigest();

	/// Return first 16 bytes of current digest from ingress mac.
	h128 ingressDigest();

protected:
	/// Update state of egress MAC with frame header.
	void updateEgressMACWithHeader(bytesConstRef _headerCipher);

	/// Update state of egress MAC with frame.
	void updateEgressMACWithFrame(bytesConstRef _cipher);
	
	/// Update state of ingress MAC with frame header.
	void updateIngressMACWithHeader(bytesConstRef _headerCipher);
	
	/// Update state of ingress MAC with frame.
	void updateIngressMACWithFrame(bytesConstRef _cipher);

private:
	/// Update state of _mac.
	void updateMAC(CryptoPP::SHA3_256& _mac, bytesConstRef _seed = bytesConstRef());

	CryptoPP::SecByteBlock m_frameEncKey;						///< Key for m_frameEnc
	CryptoPP::CTR_Mode<CryptoPP::AES>::Encryption m_frameEnc;	///< Encoder for egress plaintext.
	
	CryptoPP::SecByteBlock m_frameDecKey;						///< Key for m_frameDec
	CryptoPP::CTR_Mode<CryptoPP::AES>::Encryption m_frameDec;	///< Decoder for egress plaintext.
	
	CryptoPP::SecByteBlock m_macEncKey;						/// Key for m_macEnd
	CryptoPP::ECB_Mode<CryptoPP::AES>::Encryption m_macEnc;	/// One-way coder used by updateMAC for ingress and egress MAC updates.
	Mutex x_macEnc;											/// Mutex
	
	CryptoPP::SHA3_256 m_egressMac;			///< State of MAC for egress ciphertext.
	CryptoPP::SHA3_256 m_ingressMac;			///< State of MAC for ingress ciphertext.
};

}
}