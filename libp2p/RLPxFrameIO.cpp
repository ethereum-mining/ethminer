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
/** @file RLPXFrameIO.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#include "RLPxFrameIO.h"
#include <libdevcore/Assertions.h>
#include "Host.h"
#include "Session.h"
#include "Peer.h"
#include "RLPxHandshake.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace CryptoPP;

RLPXFrameIO::RLPXFrameIO(RLPXHandshake const& _init): m_socket(_init.m_socket)
{
	// we need:
	// originated?
	// Secret == output of ecdhe agreement
	// authCipher
	// ackCipher

	bytes keyMaterialBytes(64);
	bytesRef keyMaterial(&keyMaterialBytes);

	// shared-secret = sha3(ecdhe-shared-secret || sha3(nonce || initiator-nonce))
	Secret ephemeralShared;
	_init.m_ecdhe.agree(_init.m_remoteEphemeral, ephemeralShared);
	ephemeralShared.ref().copyTo(keyMaterial.cropped(0, h256::size));
	h512 nonceMaterial;
	h256 const& leftNonce = _init.m_originated ? _init.m_remoteNonce : _init.m_nonce;
	h256 const& rightNonce = _init.m_originated ? _init.m_nonce : _init.m_remoteNonce;
	leftNonce.ref().copyTo(nonceMaterial.ref().cropped(0, h256::size));
	rightNonce.ref().copyTo(nonceMaterial.ref().cropped(h256::size, h256::size));
	auto outRef(keyMaterial.cropped(h256::size, h256::size));
	sha3(nonceMaterial.ref(), outRef); // output h(nonces)
	
	sha3(keyMaterial, outRef); // output shared-secret
	// token: sha3(outRef, bytesRef(&token)); -> m_host (to be saved to disk)
	
	// aes-secret = sha3(ecdhe-shared-secret || shared-secret)
	sha3(keyMaterial, outRef); // output aes-secret
	m_frameEncKey.resize(h256::size);
	memcpy(m_frameEncKey.data(), outRef.data(), h256::size);
	m_frameDecKey.resize(h256::size);
	memcpy(m_frameDecKey.data(), outRef.data(), h256::size);
	h128 iv;
	m_frameEnc.SetKeyWithIV(m_frameEncKey, h256::size, iv.data());
	m_frameDec.SetKeyWithIV(m_frameDecKey, h256::size, iv.data());

	// mac-secret = sha3(ecdhe-shared-secret || aes-secret)
	sha3(keyMaterial, outRef); // output mac-secret
	m_macEncKey.resize(h256::size);
	memcpy(m_macEncKey.data(), outRef.data(), h256::size);
	m_macEnc.SetKey(m_macEncKey, h256::size);

	// Initiator egress-mac: sha3(mac-secret^recipient-nonce || auth-sent-init)
	//           ingress-mac: sha3(mac-secret^initiator-nonce || auth-recvd-ack)
	// Recipient egress-mac: sha3(mac-secret^initiator-nonce || auth-sent-ack)
	//           ingress-mac: sha3(mac-secret^recipient-nonce || auth-recvd-init)
 
	(*(h256*)outRef.data() ^ _init.m_remoteNonce).ref().copyTo(keyMaterial);
	bytes const& egressCipher = _init.m_originated ? _init.m_authCipher : _init.m_ackCipher;
	keyMaterialBytes.resize(h256::size + egressCipher.size());
	keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
	bytesConstRef(&egressCipher).copyTo(keyMaterial.cropped(h256::size, egressCipher.size()));
	m_egressMac.Update(keyMaterial.data(), keyMaterial.size());

	// recover mac-secret by re-xoring remoteNonce
	(*(h256*)keyMaterial.data() ^ _init.m_remoteNonce ^ _init.m_nonce).ref().copyTo(keyMaterial);
	bytes const& ingressCipher = _init.m_originated ? _init.m_ackCipher : _init.m_authCipher;
	keyMaterialBytes.resize(h256::size + ingressCipher.size());
	keyMaterial.retarget(keyMaterialBytes.data(), keyMaterialBytes.size());
	bytesConstRef(&ingressCipher).copyTo(keyMaterial.cropped(h256::size, ingressCipher.size()));
	m_ingressMac.Update(keyMaterial.data(), keyMaterial.size());
}

void RLPXFrameIO::writeSingleFramePacket(bytesConstRef _packet, bytes& o_bytes)
{
	// _packet = type || rlpList()

	RLPStream header;
	uint32_t len = (uint32_t)_packet.size();
	header.appendRaw(bytes({byte((len >> 16) & 0xff), byte((len >> 8) & 0xff), byte(len & 0xff)}));
	// zeroHeader: []byte{0xC2, 0x80, 0x80}. Should be rlpList(protocolType,seqId,totalPacketSize).
	header.appendRaw(bytes({0xc2,0x80,0x80}));
	
	// TODO: SECURITY check that header is <= 16 bytes

	bytes headerWithMac(h256::size);
	bytesConstRef(&header.out()).copyTo(bytesRef(&headerWithMac));
	m_frameEnc.ProcessData(headerWithMac.data(), headerWithMac.data(), 16);
	updateEgressMACWithHeader(bytesConstRef(&headerWithMac).cropped(0, 16));
	egressDigest().ref().copyTo(bytesRef(&headerWithMac).cropped(h128::size,h128::size));

	auto padding = (16 - (_packet.size() % 16)) % 16;
	o_bytes.swap(headerWithMac);
	o_bytes.resize(32 + _packet.size() + padding + h128::size);
	bytesRef packetRef(o_bytes.data() + 32, _packet.size());
	m_frameEnc.ProcessData(packetRef.data(), _packet.data(), _packet.size());
	bytesRef paddingRef(o_bytes.data() + 32 + _packet.size(), padding);
	if (padding)
		m_frameEnc.ProcessData(paddingRef.data(), paddingRef.data(), padding);
	bytesRef packetWithPaddingRef(o_bytes.data() + 32, _packet.size() + padding);
	updateEgressMACWithFrame(packetWithPaddingRef);
	bytesRef macRef(o_bytes.data() + 32 + _packet.size() + padding, h128::size);
	egressDigest().ref().copyTo(macRef);
}

bool RLPXFrameIO::authAndDecryptHeader(bytesRef io)
{
	asserts(io.size() == h256::size);
	updateIngressMACWithHeader(io);
	bytesConstRef macRef = io.cropped(h128::size, h128::size);
	h128 expected = ingressDigest();
	if (*(h128*)macRef.data() != expected)
		return false;
	m_frameDec.ProcessData(io.data(), io.data(), h128::size);
	return true;
}

bool RLPXFrameIO::authAndDecryptFrame(bytesRef io)
{
	bytesRef cipherText(io.cropped(0, io.size() - h128::size));
	updateIngressMACWithFrame(cipherText);
	bytesConstRef frameMac(io.data() + io.size() - h128::size, h128::size);
	if (*(h128*)frameMac.data() != ingressDigest())
		return false;
	m_frameDec.ProcessData(io.data(), io.data(), io.size() - h128::size);
	return true;
}

h128 RLPXFrameIO::egressDigest()
{
	SHA3_256 h(m_egressMac);
	h128 digest;
	h.TruncatedFinal(digest.data(), h128::size);
	return digest;
}

h128 RLPXFrameIO::ingressDigest()
{
	SHA3_256 h(m_ingressMac);
	h128 digest;
	h.TruncatedFinal(digest.data(), h128::size);
	return digest;
}

void RLPXFrameIO::updateEgressMACWithHeader(bytesConstRef _headerCipher)
{
	updateMAC(m_egressMac, _headerCipher.cropped(0, 16));
}

void RLPXFrameIO::updateEgressMACWithFrame(bytesConstRef _cipher)
{
	m_egressMac.Update(_cipher.data(), _cipher.size());
	updateMAC(m_egressMac);
}

void RLPXFrameIO::updateIngressMACWithHeader(bytesConstRef _headerCipher)
{
	updateMAC(m_ingressMac, _headerCipher.cropped(0, 16));
}

void RLPXFrameIO::updateIngressMACWithFrame(bytesConstRef _cipher)
{
	m_ingressMac.Update(_cipher.data(), _cipher.size());
	updateMAC(m_ingressMac);
}

void RLPXFrameIO::updateMAC(SHA3_256& _mac, bytesConstRef _seed)
{
	if (_seed.size() && _seed.size() != h128::size)
		asserts(false);

	SHA3_256 prevDigest(_mac);
	h128 encDigest(h128::size);
	prevDigest.TruncatedFinal(encDigest.data(), h128::size);
	h128 prevDigestOut = encDigest;

	{
		Guard l(x_macEnc);
		m_macEnc.ProcessData(encDigest.data(), encDigest.data(), 16);
	}
	if (_seed.size())
		encDigest ^= *(h128*)_seed.data();
	else
		encDigest ^= *(h128*)prevDigestOut.data();

	// update mac for final digest
	_mac.Update(encDigest.data(), h128::size);
}
