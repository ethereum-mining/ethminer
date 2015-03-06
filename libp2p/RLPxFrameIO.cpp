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

#include "Host.h"
#include "Session.h"
#include "Peer.h"
#include "RLPxHandshake.h"
#include "RLPxFrameIO.h"
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
	m_frameEnc.SetKeyWithIV(outRef.data(), h128::size, h128().data());
	m_frameDec.SetKeyWithIV(outRef.data(), h128::size, h128().data());

	// mac-secret = sha3(ecdhe-shared-secret || aes-secret)
	sha3(keyMaterial, outRef); // output mac-secret
	m_macEnc.SetKey(outRef.data(), h128::size);

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

	// current/old packet format: prep(_s).appendList(_args + 1).append((unsigned)_id);
	RLPStream header;
	header.appendRaw(bytes({byte(_packet.size() >> 16), byte(_packet.size() >> 8), byte(_packet.size())}));
	// zeroHeader: []byte{0xC2, 0x80, 0x80}. Should be rlpList(protocolType,seqId,totalPacketSize).
	header.appendRaw(bytes({0xc2,0x80,0x80}));
	
	// TODO: SECURITY check that header is <= 16 bytes
	
	bytes headerWithMac;
	header.swapOut(headerWithMac);
	headerWithMac.resize(32);
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
	updateEgressMACWithEndOfFrame(packetWithPaddingRef);
	bytesRef macRef(o_bytes.data() + 32 + _packet.size() + padding, h128::size);
	egressDigest().ref().copyTo(macRef);
	clog(NetConnect) << "SENT FRAME " << _packet.size() << *(h128*)macRef.data();
	clog(NetConnect) << "FRAME TAIL " << *(h128*)(o_bytes.data() + 32 + _packet.size() + padding);
}

bool RLPXFrameIO::authAndDecryptHeader(h256& io)
{
	updateIngressMACWithHeader(io.ref());
	bytesConstRef macRef = io.ref().cropped(h128::size, h128::size);
	if (*(h128*)macRef.data() != ingressDigest())
		return false;
	m_frameDec.ProcessData(io.data(), io.data(), 16);
	return true;
}

bool RLPXFrameIO::authAndDecryptFrame(bytesRef io)
{
	bytesRef cipherText(io.cropped(0, io.size() - h128::size));
	updateIngressMACWithEndOfFrame(cipherText);
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
	return move(digest);
}

h128 RLPXFrameIO::ingressDigest()
{
	SHA3_256 h(m_ingressMac);
	h128 digest;
	h.TruncatedFinal(digest.data(), h128::size);
	return move(digest);
}

void RLPXFrameIO::updateEgressMACWithHeader(bytesConstRef _headerCipher)
{
	updateMAC(m_egressMac, *(h128*)_headerCipher.data());
}

void RLPXFrameIO::updateEgressMACWithEndOfFrame(bytesConstRef _cipher)
{
	m_egressMac.Update(_cipher.data(), _cipher.size());
	updateMAC(m_egressMac);
	{
		SHA3_256 prev(m_egressMac);
		h128 digest;
		prev.TruncatedFinal(digest.data(), h128::size);
		clog(NetConnect) << "EGRESS FRAMEMAC " << _cipher.size() << digest;
	}
}

void RLPXFrameIO::updateIngressMACWithHeader(bytesConstRef _headerCipher)
{
	updateMAC(m_ingressMac, *(h128*)_headerCipher.data());
}

void RLPXFrameIO::updateIngressMACWithEndOfFrame(bytesConstRef _cipher)
{
	m_ingressMac.Update(_cipher.data(), _cipher.size());
	updateMAC(m_ingressMac);
	{
		SHA3_256 prev(m_ingressMac);
		h128 digest;
		prev.TruncatedFinal(digest.data(), h128::size);
		clog(NetConnect) << "INGRESS FRAMEMAC " << _cipher.size() << digest;
	}
}

void RLPXFrameIO::updateMAC(SHA3_256& _mac, h128 const& _seed)
{
	SHA3_256 prevDigest(_mac);
	h128 prevDigestOut;
	prevDigest.TruncatedFinal(prevDigestOut.data(), h128::size);
	
	h128 encDigest;
	m_macEnc.ProcessData(encDigest.data(), prevDigestOut.data(), h128::size);
	encDigest ^= (!!_seed ? _seed : prevDigestOut);
	
	// update mac for final digest
	_mac.Update(encDigest.data(), h128::size);
}
