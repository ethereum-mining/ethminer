#pragma once

namespace dev
{
namespace eth
{

class MinerHook
{
public:
	virtual ~MinerHook() = default;

	// reports progress, return true to abort
	virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
	virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
};

}
}