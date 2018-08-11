/// ethminer -- Ethereum miner with OpenCL, CUDA and stratum support.
/// Copyright 2018 ethminer Authors.
/// Licensed under GNU General Public License, Version 3. See the LICENSE file.

#pragma once

#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>

#include "Exceptions.h"

namespace dev
{
namespace eth
{
/// An Ethereum address: 20 bytes.
using Address = h160;

/// The log bloom's size (2048-bit).
using LogBloom = h2048;

using BlockNumber = unsigned;


/** @brief Encapsulation of a block header.
 * Class to contain all of a block header's data. It is able to parse a block header and populate
 * from some given RLP block serialisation with the static fromHeader(), through the method
 * populateFromHeader(). This will conduct a minimal level of verification. In this case extra
 * verification can be performed through verifyInternals() and verifyParent().
 *
 * The object may also be populated from an entire block through the explicit
 * constructor BlockInfo(bytesConstRef) and manually with the populate() method. These will
 * conduct verification of the header against the other information in the block.
 *
 * The object may be populated with a template given a parent BlockInfo object with the
 * populateFromParent() method. The genesis block info may be retrieved with genesis() and the
 * corresponding RLP block created with createGenesisBlock().
 *
 * The difficulty and gas-limit derivations may be calculated with the calculateDifficulty()
 * and calculateGasLimit() and the object serialised to RLP with streamRLP. To determine the
 * header hash without the nonce (for mining), the method headerHash(WithoutNonce) is provided.
 *
 * The default constructor creates an empty object, which can be tested against with the boolean
 * conversion operator.
 */
class BlockHeader
{
public:
    static const unsigned BasicFields = 13;

    BlockHeader() = default;

    explicit operator bool() const { return m_timestamp != Invalid256; }

    h256 const& boundary() const;

    void setNumber(u256 const& _v)
    {
        m_number = _v;
        noteDirty();
    }

    void setDifficulty(u256 const& _v)
    {
        m_difficulty = _v;
        noteDirty();
    }

    u256 const& number() const { return m_number; }

    /// sha3 of the header only.
    h256 const& hashWithout() const;

    void noteDirty() const { m_hashWithout = m_boundary = h256(); }

    uint64_t nonce() const { return m_nonce; }

private:
    void streamRLPFields(RLPStream& _s) const;

    h256 m_parentHash;
    h256 m_sha3Uncles;
    Address m_coinbaseAddress;
    h256 m_stateRoot;
    h256 m_transactionsRoot;
    h256 m_receiptsRoot;
    LogBloom m_logBloom;
    u256 m_number;
    u256 m_gasLimit;
    u256 m_gasUsed;
    u256 m_timestamp = Invalid256;
    bytes m_extraData;

    u256 m_difficulty;

    mutable h256 m_hashWithout;  ///< SHA3 hash of the block header! Not serialised.
    mutable h256 m_boundary;     ///< 2^256 / difficulty

    uint64_t m_nonce = 0;
};

}  // namespace eth
}  // namespace dev
