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
/** @file RLPXFrameIO.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */


#pragma once

#include <memory>
#include <libdevcrypto/Common.h>
#include <libdevcrypto/ECDHE.h>
#include <libdevcrypto/CryptoPP.h>
#include <libdevcore/Guards.h>
#include "Common.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace dev
{
namespace p2p
{
	
class RLPXHandshake;

/**
 * @brief Encoder/decoder transport for RLPx connections established by RLPXHandshake.
 * Managed (via shared_ptr) socket for use by RLPXHandshake and RLPXFrameIO.
 *
 * Thread Safety
 * Distinct Objects: Safe.
 * Shared objects: Unsafe.
 * * an instance method must not be called concurrently
 * * a writeSingleFramePacket can be called concurrent to authAndDecryptHeader OR authAndDecryptFrame
 */
class RLPXSocket: public std::enable_shared_from_this<RLPXSocket>
{
public:
	RLPXSocket(bi::tcp::socket* _socket): m_socket(std::move(*_socket)) {}
	~RLPXSocket() { close(); }
	
	bool isConnected() const { return m_socket.is_open(); }
	void close() { try { boost::system::error_code ec; m_socket.shutdown(bi::tcp::socket::shutdown_both, ec); if (m_socket.is_open()) m_socket.close(); } catch (...){} }
	bi::tcp::endpoint remoteEndpoint() { try { return m_socket.remote_endpoint(); } catch (...){ return bi::tcp::endpoint(); } }
	bi::tcp::socket& ref() { return m_socket; }
	
protected:
	bi::tcp::socket m_socket;
};

/**
 * @brief Encoder/decoder transport for RLPx connections established by RLPXHandshake.
 *
 * Thread Safety 
 * Distinct Objects: Safe.
 * Shared objects: Unsafe.
 */
class RLPXFrameIO
{
	friend class Session;
public:
	/// Constructor.
	/// Requires instance of RLPXHandshake which has completed first two phases of handshake.
	RLPXFrameIO(RLPXHandshake const& _init);
	~RLPXFrameIO() {}
	
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
	
	bi::tcp::socket& socket() { return m_socket->ref(); }
	
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
	
	std::shared_ptr<RLPXSocket> m_socket;
};

}
}