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

void RLPXFrameWriter::enque(unsigned _packetType, RLPStream& _payload, PacketPriority _priority)
{
	QueueState& qs = _priority ? m_q.first : m_q.second;
	DEV_GUARDED(qs.x)
		qs.q.push_back(move(RLPXPacket(m_protocolType, _packetType, _payload)));
}

size_t RLPXFrameWriter::drain(RLPXFrameCoder& _coder, unsigned _size, vector<bytes>& o_toWrite)
{
	static const size_t c_blockSize = h128::size;
	static const size_t c_overhead = c_blockSize * 2;
	if (_size < c_overhead + c_blockSize)
		return 0;

	size_t ret = 0;
	size_t frameLen = _size;
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
			return 0;

		// first run when !swapQueues, high > low, otherwise low > high
		bool high = highPending && !swapQueues ? true : lowPending ? false : true;
		QueueState &qs = high ? m_q.first : m_q.second;
		size_t frameAllot = (!swapQueues && highPending && lowPending ? frameLen / 2 - (c_overhead + c_blockSize) > 0 ? frameLen / 2 : frameLen : frameLen) - c_overhead;
		size_t offset;
		while (frameAllot >= c_blockSize)
		{
			if (qs.writing == nullptr)
			{
				DEV_GUARDED(qs.x)
					qs.writing = &qs.q[0];
				qs.remaining = qs.writing->size();
				bytes packetType = rlp(qs.writing->type());
				qs.sequenced = qs.remaining + packetType.size() > frameAllot;

				// stop here if we can't write-out packet-type
				// or payload already packed and packet won't fit
				if (packetType.size() > frameAllot || (qs.sequenced && payload.size()))
				{
					qs.writing = nullptr;
					qs.remaining = 0;
					qs.sequenced = false;
					break;
				}
				else if (qs.sequenced)
					qs.sequence = ++m_sequenceId;

				frameAllot -= packetType.size();
				payload += packetType;
			}
			assert(qs.sequenced || (!qs.sequenced && frameAllot >= qs.remaining));
			if (frameAllot && qs.remaining)
			{
				offset = qs.writing->size() - qs.remaining;
				auto length = qs.remaining <= frameAllot ? qs.remaining : frameAllot;
				bytes portion = bytesConstRef(&qs.writing->data()).cropped(offset, length).toBytes();
				qs.remaining -= length;
				frameAllot -= portion.size();
				payload += portion;
			}
			if (!qs.remaining)
				qs.writing = nullptr;
			if (qs.sequenced)
				break;
		}
		
		if (payload.size())
		{
			if (qs.sequenced)
				if (!offset)
					_coder.writeFrame(m_protocolType, qs.sequence, qs.writing->size(), &payload, payload);
				else
					_coder.writeFrame(m_protocolType, qs.sequence, &payload, payload);
			else
				_coder.writeFrame(m_protocolType, &payload, payload);
			frameLen -= payload.size();
			o_toWrite.push_back(move(payload));
			payload.resize(0);
			
			if (!qs.remaining)
			{
				qs.writing = nullptr;
				qs.sequenced = false;
				DEV_GUARDED(qs.x)
					qs.q.pop_front();
				ret++;
			}
		}
		else if (swapQueues)
			break;
		swapQueues = true;
	}
	return ret;
}
