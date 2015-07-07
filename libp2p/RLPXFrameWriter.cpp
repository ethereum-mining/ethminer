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
/** @file RLPXFrameWriter.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#include "RLPXFrameWriter.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;

const uint16_t RLPXFrameWriter::EmptyFrameLength = h128::size * 3; // header + headerMAC + frameMAC
const uint16_t RLPXFrameWriter::MinFrameDequeLength = h128::size * 4; // header + headerMAC + padded-block + frameMAC

void RLPXFrameWriter::enque(RLPXPacket&& _p, PacketPriority _priority)
{
	if (!_p.isValid())
		return;
	WriterState& qs = _priority ? m_q.first : m_q.second;
	DEV_GUARDED(qs.x)
		qs.q.push_back(move(_p));
}

void RLPXFrameWriter::enque(uint8_t _packetType, RLPStream& _payload, PacketPriority _priority)
{
	enque(RLPXPacket(m_protocolType, (RLPStream() << _packetType), _payload), _priority);
}

size_t RLPXFrameWriter::mux(RLPXFrameCoder& _coder, unsigned _size, vector<bytes>& o_toWrite)
{
	static const size_t c_blockSize = h128::size;
	static const size_t c_overhead = c_blockSize * 3; // header + headerMac + frameMAC
	if (_size < c_overhead + c_blockSize)
		return 0;

	size_t ret = 0;
	size_t frameLen = _size / 16 * 16;
	bytes payload(0);
	bool swapQueues = false;
	while (frameLen >= c_overhead + c_blockSize)
	{
		bool highPending;
		bool lowPending;
		DEV_GUARDED(m_q.first.x)
			highPending = !!m_q.first.q.size();
		DEV_GUARDED(m_q.second.x)
			lowPending = !!m_q.second.q.size();

		if (!highPending && !lowPending)
			return ret;
		
		// first run when !swapQueues, high > low, otherwise low > high
		bool high = highPending && !swapQueues ? true : lowPending ? false : true;
		WriterState &qs = high ? m_q.first : m_q.second;
		size_t frameAllot = (!swapQueues && highPending && lowPending ? frameLen / 2 - (c_overhead + c_blockSize) > 0 ? frameLen / 2 : frameLen : frameLen) - c_overhead;
		size_t offset = 0;
		size_t length = 0;
		while (frameAllot >= c_blockSize)
		{
			// Invariants:
			// !qs.writing && payload.empty()	initial entry
			// !qs.writing && !payload.empty()	continuation (multiple packets per frame)
			// qs.writing && payload.empty()	initial entry, continuation (multiple frames per packet)
			// qs.writing && !payload.empty()	INVALID
			
			// write packet-type
			if (qs.writing == nullptr)
			{
				{
					Guard l(qs.x);
					if (!qs.q.empty())
						qs.writing = &qs.q[0];
					else
						break;
				}

				// break here if we can't write-out packet-type
				// or payload is packed and next packet won't fit (implicit)
				qs.multiFrame = qs.writing->size() > frameAllot;
				assert(qs.writing->type().size());
				if (qs.writing->type().size() > frameAllot || (qs.multiFrame && !payload.empty()))
				{
					qs.writing = nullptr;
					qs.remaining = 0;
					qs.multiFrame = false;
					break;
				}
				else if (qs.multiFrame)
					qs.sequence = ++m_sequenceId;
				
				frameAllot -= qs.writing->type().size();
				payload += qs.writing->type();
				
				qs.remaining = qs.writing->data().size();
			}
			
			// write payload w/remaining allotment
			assert(qs.multiFrame || (!qs.multiFrame && frameAllot >= qs.remaining));
			if (frameAllot && qs.remaining)
			{
				offset = qs.writing->data().size() - qs.remaining;
				length = qs.remaining <= frameAllot ? qs.remaining : frameAllot;
				bytes portion = bytesConstRef(&qs.writing->data()).cropped(offset, length).toBytes();
				qs.remaining -= length;
				frameAllot -= portion.size();
				payload += portion;
			}
			
			assert((!qs.remaining && (offset > 0 || !qs.multiFrame)) || (qs.remaining && qs.multiFrame));
			if (!qs.remaining)
			{
				qs.writing = nullptr;
				DEV_GUARDED(qs.x)
					qs.q.pop_front();
				ret++;
			}
			// qs.writing is left alone for first frame of multi-frame packet
			if (qs.multiFrame)
				break;
		}
		
		if (!payload.empty())
		{
			if (qs.multiFrame)
				if (offset == 0)
					// 1st frame of segmented packet writes total-size of packet
					_coder.writeFrame(m_protocolType, qs.sequence, qs.writing->size(), &payload, payload);
				else
					_coder.writeFrame(m_protocolType, qs.sequence, &payload, payload);
			else
				_coder.writeFrame(m_protocolType, &payload, payload);
			assert((int)frameLen - payload.size() >= 0);
			frameLen -= payload.size();
			o_toWrite.push_back(payload);
			payload.resize(0);
			
			if (!qs.remaining && qs.multiFrame)
				qs.multiFrame = false;
		}
		else if (swapQueues)
			break;
		swapQueues = true;
	}
	return ret;
}
