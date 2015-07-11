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

struct RLPXFrameDecryptFailed: virtual dev::Exception {};

/**
 * @brief Encapsulation of Frame
 * @todo coder integration; padding derived from coder
 */
struct RLPXFrameInfo
{
	RLPXFrameInfo() = default;
	/// Constructor. frame-size || protocol-type, [sequence-id[, total-packet-size]]
	RLPXFrameInfo(bytesConstRef _frameHeader);

	uint32_t const length;			///< Size of frame (excludes padding). Max: 2**24
	uint8_t const padding;			///< Length of padding which follows @length.
	
	bytes const data;				///< Bytes of Header.
	RLP const header;				///< Header RLP.
	
	uint16_t const protocolId;		///< Protocol ID as negotiated by handshake.
	bool const multiFrame;			///< If this frame is part of a sequence
	uint16_t const sequenceId;		///< Sequence ID of frame
	uint32_t const totalLength;		///< Set to total length of packet in first frame of multiframe packet
};

class RLPXHandshake;

/**
 * @brief Encoder/decoder transport for RLPx connection established by RLPXHandshake.
 *
 * @todo rename to RLPXTranscoder
 * @todo Remove 'Frame' nomenclature and expect caller to provide RLPXFrame
 * @todo Remove handshake as friend, remove handshake-based constructor
 *
 * Thread Safety 
 * Distinct Objects: Unsafe.
 * Shared objects: Unsafe.
 */
class RLPXFrameCoder
{
	friend class RLPXFrameIOMux;
	friend class Session;
public:
	/// Construct; requires instance of RLPXHandshake which has encrypted ECDH key exchange (first two phases of handshake).
	RLPXFrameCoder(RLPXHandshake const& _init);
	
	/// Construct with external key material.
	RLPXFrameCoder(bool _originated, h512 const& _remoteEphemeral, h256 const& _remoteNonce, crypto::ECDHE const& _ephemeral, h256 const& _nonce, bytesConstRef _ackCipher, bytesConstRef _authCipher);
	
	~RLPXFrameCoder() {}
	
	/// Establish shared secrets and setup AES and MAC states.
	void setup(bool _originated, h512 const& _remoteEphemeral, h256 const& _remoteNonce, crypto::ECDHE const& _ephemeral, h256 const& _nonce, bytesConstRef _ackCipher, bytesConstRef _authCipher);
	
	/// Write single-frame payload of packet(s).
	void writeFrame(uint16_t _protocolType, bytesConstRef _payload, bytes& o_bytes);

	/// Write continuation frame of segmented payload.
	void writeFrame(uint16_t _protocolType, uint16_t _seqId, bytesConstRef _payload, bytes& o_bytes);
	
	/// Write first frame of segmented or sequence-tagged payload.
	void writeFrame(uint16_t _protocolType, uint16_t _seqId, uint32_t _totalSize, bytesConstRef _payload, bytes& o_bytes);
	
	/// Legacy. Encrypt _packet as ill-defined legacy RLPx frame.
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
	void writeFrame(RLPStream const& _header, bytesConstRef _payload, bytes& o_bytes);
	
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