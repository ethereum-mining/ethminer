#include <dbus/dbus.h>

using namespace std;

class DBusInt
{
public:
		DBusInt()
		{
			dbus_error_init(&err);
			conn = dbus_bus_get(DBUS_BUS_SESSION, &err);
			if (!conn)
			{
				cerr << "DBus error " << err.name << ": " << err.message << endl;
			}
			dbus_bus_request_name(conn, "eth.miner", DBUS_NAME_FLAG_REPLACE_EXISTING, &err);
			if (dbus_error_is_set(&err))
			{
				cerr << "DBus error " << err.name << ": " << err.message << endl;
				dbus_connection_close(conn);
			}
			cout << "DBus initialized!" << endl;
		}

		void send(const char *hash)
		{
			DBusMessage *msg;
			msg = dbus_message_new_signal("/eth/miner/hash", "eth.miner.monitor", "Hash");
			if (msg == nullptr)
			{
				cerr << "Message is null!" << endl;
			}
			dbus_message_append_args(msg, DBUS_TYPE_STRING, &hash, DBUS_TYPE_INVALID);
			if (!dbus_connection_send(conn, msg, NULL)) cerr << "Error sending message!" << endl;
			dbus_message_unref(msg);
		}

private:
		DBusError err;
		DBusConnection *conn;
};
