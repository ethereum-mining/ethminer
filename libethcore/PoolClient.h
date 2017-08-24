class PoolClient
{
public:
	PoolClient(string const & host, string const & port, string const & user, string const & pass);
	~PoolClient();
	
	void setFailover(string const & host, string const & port, string const & user, string const & pass);
	
	void connect();
	void reconnect();
	void disconnect();
	
	bool submitHashrate(string const & rate);
	bool submitSolution(Solution solution);
	
	bool isRunning() { return m_running; }
	bool isConnected() { return m_connected && m_authorized; }
	
	using SolutionAccepted = std::function<void(boolean const&)>;
	using SolutionRejected = std::function<void(boolean const&)>;
	using Disconnected = std::function<void()>;
	using Connected = std::function<void()>;
	
	void onSolutionAccepted(SolutionAccepted const& _handler) ) { m_onSolutionAccepted = _handler; }
	void onSolutionRejected(SolutionRejected const& _handler) ) { m_onSolutionRejected = _handler; }
	void onDisconnected(Disconnected const& _handler) ) { m_onDisconnected = _handler; }
	void onConnected(Connected const& _handler) ) { m_onConnected = _handler; }
		
private:
	bool m_authorized;
	bool m_connected;
	bool m_running = true;
	
	SolutionAccepted m_onSolutionAccepted;
	SolutionRejected m_onSolutionRejected;
	Disconnected m_onDisconnected;
	Connected m_onConnected;
}