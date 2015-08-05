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
/** @file Transact.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

// Make sure boost/asio.hpp is included before windows.h.
#include <boost/asio.hpp>

#include "Transact.h"

#include <fstream>
#include <boost/algorithm/string.hpp>
#include <QFileDialog>
#include <QMessageBox>
#include <QClipboard>
#include <liblll/Compiler.h>
#include <liblll/CodeFragment.h>
#if ETH_SOLIDITY || !ETH_TRUE
#include <libsolidity/CompilerStack.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/AST.h>
#include <libsolidity/SourceReferenceFormatter.h>
#endif
#include <libnatspec/NatspecExpressionEvaluator.h>
#include <libethereum/Client.h>
#include <libethereum/Utility.h>
#include <libethcore/KeyManager.h>

#if ETH_SERPENT
#include <libserpent/funcs.h>
#include <libserpent/util.h>
#endif
#include "Debugger.h"
#include "ui_Transact.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

Transact::Transact(Context* _c, QWidget* _parent):
	QDialog(_parent),
	ui(new Ui::Transact),
	m_context(_c)
{
	ui->setupUi(this);

	resetGasPrice();
	setValueUnits(ui->valueUnits, ui->value, 0);

	on_destination_currentTextChanged(QString());
}

Transact::~Transact()
{
	delete ui;
}

void Transact::setEnvironment(AddressHash const& _accounts, dev::eth::Client* _eth, NatSpecFace* _natSpecDB)
{
	m_accounts = _accounts;
	m_ethereum = _eth;
	m_natSpecDB = _natSpecDB;

	auto old = ui->from->currentIndex();
	ui->from->clear();
	for (auto const& address: m_accounts)
	{
		u256 b = ethereum()->balanceAt(address, PendingBlock);
		QString s = QString("%4 %2: %1").arg(formatBalance(b).c_str()).arg(QString::fromStdString(m_context->render(address))).arg(QString::fromStdString(m_context->keyManager().accountName(address)));
		ui->from->addItem(s);
	}
	if (old > -1 && old < ui->from->count())
		ui->from->setCurrentIndex(old);
	else if (ui->from->count())
		ui->from->setCurrentIndex(0);
}

void Transact::resetGasPrice()
{
	setValueUnits(ui->gasPriceUnits, ui->gasPrice, m_context->gasPrice());
}

bool Transact::isCreation() const
{
	return ui->destination->currentText().isEmpty() || ui->destination->currentText() == "(Create Contract)";
}

u256 Transact::fee() const
{
	return ui->gas->value() * gasPrice();
}

u256 Transact::value() const
{
	if (ui->valueUnits->currentIndex() == -1)
		return 0;
	return ui->value->value() * units()[units().size() - 1 - ui->valueUnits->currentIndex()].first;
}

u256 Transact::gasPrice() const
{
	if (ui->gasPriceUnits->currentIndex() == -1)
		return 0;
	return ui->gasPrice->value() * units()[units().size() - 1 - ui->gasPriceUnits->currentIndex()].first;
}

Address Transact::to() const
{
	return m_context->fromString(ui->destination->currentText().toStdString()).first;
}

u256 Transact::total() const
{
	return value() + fee();
}

void Transact::updateDestination()
{
	cwatch << "updateDestination()";
	QString s;
	for (auto i: ethereum()->addresses())
		if ((s = QString::fromStdString(m_context->pretty(i))).size())
			// A namereg address
			if (ui->destination->findText(s, Qt::MatchExactly | Qt::MatchCaseSensitive) == -1)
				ui->destination->addItem(s);
	for (int i = 0; i < ui->destination->count(); ++i)
		if (ui->destination->itemText(i) != "(Create Contract)" && !to())
			ui->destination->removeItem(i--);
}

void Transact::updateFee()
{
//	ui->fee->setText(QString("(gas sub-total: %1)").arg(formatBalance(fee()).c_str()));
	auto totalReq = total();
	ui->total->setText(QString("Total: %1").arg(formatBalance(totalReq).c_str()));

	bool ok = false;
	for (auto const& i: m_accounts)
		if (ethereum()->balanceAt(i) >= totalReq)
		{
			ok = true;
			break;
		}
//	ui->send->setEnabled(ok);
	QPalette p = ui->total->palette();
	p.setColor(QPalette::WindowText, QColor(ok ? 0x00 : 0x80, 0x00, 0x00));
	ui->total->setPalette(p);
}

void Transact::on_destination_currentTextChanged(QString)
{
	if (ui->destination->currentText().size() && ui->destination->currentText() != "(Create Contract)")
	{
		auto p = m_context->fromString(ui->destination->currentText().toStdString());
		if (p.first)
			ui->calculatedName->setText(QString::fromStdString(m_context->render(p.first)));
		else
			ui->calculatedName->setText("Unknown Address");
		if (!p.second.empty())
		{
			m_data = p.second;
			ui->data->setPlainText(QString::fromStdString("0x" + toHex(m_data)));
			ui->data->setEnabled(false);
		}
		else if (!ui->data->isEnabled())
		{
			m_data.clear();
			ui->data->setPlainText("");
			ui->data->setEnabled(true);
		}
	}
	else
		ui->calculatedName->setText("Create Contract");
	rejigData();
	//	updateFee();
}

void Transact::on_copyUnsigned_clicked()
{
	auto a = fromAccount();
	u256 nonce = ui->autoNonce->isChecked() ? ethereum()->countAt(a, PendingBlock) : ui->nonce->value();

	Transaction t;
	if (isCreation())
		// If execution is a contract creation, add Natspec to
		// a local Natspec LEVELDB
		t = Transaction(value(), gasPrice(), ui->gas->value(), m_data, nonce);
	else
		// TODO: cache like m_data.
		t = Transaction(value(), gasPrice(), ui->gas->value(), to(), m_data, nonce);
	qApp->clipboard()->setText(QString::fromStdString(toHex(t.rlp())));
}

static std::string toString(TransactionException _te)
{
	switch (_te)
	{
	case TransactionException::Unknown: return "Unknown error";
	case TransactionException::InvalidSignature: return "Permanent Abort: Invalid transaction signature";
	case TransactionException::InvalidNonce: return "Transient Abort: Invalid transaction nonce";
	case TransactionException::NotEnoughCash: return "Transient Abort: Not enough cash to pay for transaction";
	case TransactionException::OutOfGasBase: return "Permanent Abort: Not enough gas to consider transaction";
	case TransactionException::BlockGasLimitReached: return "Transient Abort: Gas limit of block reached";
	case TransactionException::BadInstruction: return "VM Error: Attempt to execute invalid instruction";
	case TransactionException::BadJumpDestination: return "VM Error: Attempt to jump to invalid destination";
	case TransactionException::OutOfGas: return "VM Error: Out of gas";
	case TransactionException::OutOfStack: return "VM Error: VM stack limit reached during execution";
	case TransactionException::StackUnderflow: return "VM Error: Stack underflow";
	default:; return std::string();
	}
}

#if ETH_SOLIDITY
static string getFunctionHashes(dev::solidity::CompilerStack const& _compiler, string const& _contractName)
{
	string ret = "";
	auto const& contract = _compiler.getContractDefinition(_contractName);
	auto interfaceFunctions = contract.getInterfaceFunctions();

	for (auto const& it: interfaceFunctions)
	{
		ret += it.first.abridged();
		ret += " :";
		ret += it.second->getDeclaration().getName() + "\n";
	}
	return ret;
}
#endif

static tuple<vector<string>, bytes, string> userInputToCode(string const& _user, bool _opt)
{
	string lll;
	string solidity;
	bytes data;
	vector<string> errors;
	if (_user.find_first_not_of("1234567890abcdefABCDEF\n\t ") == string::npos && _user.size() % 2 == 0)
	{
		std::string u = _user;
		boost::replace_all_copy(u, "\n", "");
		boost::replace_all_copy(u, "\t", "");
		boost::replace_all_copy(u, " ", "");
		data = fromHex(u);
	}
#if ETH_SOLIDITY || !ETH_TRUE
	else if (sourceIsSolidity(_user))
	{
		dev::solidity::CompilerStack compiler(true);
		try
		{
//				compiler.addSources(dev::solidity::StandardSources);
			data = compiler.compile(_user, _opt);
			solidity = "<h4>Solidity</h4>";
			solidity += "<pre>var " + compiler.defaultContractName() + " = web3.eth.contract(" + QString::fromStdString(compiler.getInterface()).replace(QRegExp("\\s"), "").toHtmlEscaped().toStdString() + ");</pre>";
			solidity += "<pre>" + QString::fromStdString(compiler.getSolidityInterface()).toHtmlEscaped().toStdString() + "</pre>";
			solidity += "<pre>" + QString::fromStdString(getFunctionHashes(compiler, "")).toHtmlEscaped().toStdString() + "</pre>";
		}
		catch (dev::Exception const& exception)
		{
			ostringstream error;
			solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler);
			errors.push_back("Solidity: " + error.str());
		}
		catch (...)
		{
			errors.push_back("Solidity: Uncaught exception");
		}
	}
#endif
#if ETH_SERPENT
	else if (sourceIsSerpent(_user))
	{
		try
		{
			data = dev::asBytes(::compile(_user));
		}
		catch (string const& err)
		{
			errors.push_back("Serpent " + err);
		}
	}
#endif
	else
	{
		data = compileLLL(_user, _opt, &errors);
		if (errors.empty())
		{
			auto asmcode = compileLLLToAsm(_user, _opt);
			lll = "<h4>LLL</h4><pre>" + QString::fromStdString(asmcode).toHtmlEscaped().toStdString() + "</pre>";
		}
	}
	return make_tuple(errors, data, lll + solidity);
}

string Transact::natspecNotice(Address _to, bytes const& _data)
{
	if (ethereum()->codeAt(_to, PendingBlock).size())
	{
		string userNotice = m_natSpecDB->getUserNotice(ethereum()->postState().codeHash(_to), _data);
		if (userNotice.empty())
			return "Destination contract unknown.";
		else
		{
			NatspecExpressionEvaluator evaluator;
			return evaluator.evalExpression(QString::fromStdString(userNotice)).toStdString();
		}
	}
	else
		return "Destination not a contract.";
}

void Transact::rejigData()
{
	if (!ethereum())
		return;

	// Determine how much balance we have to play with...
	//findSecret(value() + ethereum()->gasLimitRemaining() * gasPrice());
	auto s = fromAccount();
	if (!s)
		return;

	auto b = ethereum()->balanceAt(s, PendingBlock);

	m_allGood = true;
	QString htmlInfo;

	auto bail = [&](QString he) {
		m_allGood = false;
//		ui->send->setEnabled(false);
		ui->code->setHtml(he + htmlInfo);
	};

	// Determine m_info.
	if (isCreation())
	{
		string info;
		vector<string> errors;
		tie(errors, m_data, info) = userInputToCode(ui->data->toPlainText().toStdString(), ui->optimize->isChecked());
		if (errors.size())
		{
			// Errors determining transaction data (i.e. init code). Bail.
			QString htmlErrors;
			for (auto const& i: errors)
				htmlErrors.append("<div class=\"error\"><span class=\"icon\">ERROR</span> " + QString::fromStdString(i).toHtmlEscaped() + "</div>");
			bail(htmlErrors);
			return;
		}
		htmlInfo = QString::fromStdString(info) + "<h4>Code</h4>" + QString::fromStdString(disassemble(m_data)).toHtmlEscaped();
	}
	else
	{
		m_data = parseData(ui->data->toPlainText().toStdString());
		htmlInfo = "<h4>Dump</h4>" + QString::fromStdString(dev::memDump(m_data, 8, true));
	}

	htmlInfo += "<h4>Hex</h4>" + QString(ETH_HTML_DIV(ETH_HTML_MONO)) + QString::fromStdString(toHex(m_data)) + "</div>";

	// Determine the minimum amount of gas we need to play...
	qint64 baseGas = (qint64)Transaction::gasRequired(m_data, 0);
	qint64 gasNeeded = 0;

	if (b < value() + baseGas * gasPrice())
	{
		// Not enough - bail.
		bail("<div class=\"error\"><span class=\"icon\">ERROR</span> Account doesn't contain enough for paying even the basic amount of gas required.</div>");
		return;
	}
	else
		gasNeeded = (qint64)min<bigint>(ethereum()->gasLimitRemaining(), ((b - value()) / max<u256>(gasPrice(), 1)));

	// Dry-run execution to determine gas requirement and any execution errors
	Address to;
	ExecutionResult er;
	if (isCreation())
		er = ethereum()->create(s, value(), m_data, gasNeeded, gasPrice());
	else
	{
		// TODO: cache like m_data.
		to = m_context->fromString(ui->destination->currentText().toStdString()).first;
		er = ethereum()->call(s, value(), to, m_data, gasNeeded, gasPrice());
	}
	gasNeeded = (qint64)(er.gasUsed + er.gasRefunded + c_callStipend);
	htmlInfo = QString("<div class=\"info\"><span class=\"icon\">INFO</span> Gas required: %1 total = %2 base, %3 exec [%4 refunded later]</div>").arg(gasNeeded).arg(baseGas).arg(gasNeeded - baseGas).arg((qint64)er.gasRefunded) + htmlInfo;

	if (er.excepted != TransactionException::None)
	{
		bail("<div class=\"error\"><span class=\"icon\">ERROR</span> " + QString::fromStdString(toString(er.excepted)) + "</div>");
		return;
	}
	if (er.codeDeposit == CodeDeposit::Failed)
	{
		bail("<div class=\"error\"><span class=\"icon\">ERROR</span> Code deposit failed due to insufficient gas; " + QString::fromStdString(toString(er.gasForDeposit)) + " GAS &lt; " + QString::fromStdString(toString(er.depositSize)) + " bytes * " + QString::fromStdString(toString(c_createDataGas)) + "GAS/byte</div>");
		return;
	}

	// Add Natspec information
	if (!isCreation())
		htmlInfo = "<div class=\"info\"><span class=\"icon\">INFO</span> " + QString::fromStdString(natspecNotice(to, m_data)).toHtmlEscaped() + "</div>" + htmlInfo;

	// Update gas
	if (ui->gas->value() == ui->gas->minimum())
	{
		ui->gas->setMinimum(gasNeeded);
		ui->gas->setValue(gasNeeded);
	}
	else
		ui->gas->setMinimum(gasNeeded);

	updateFee();

	ui->code->setHtml(htmlInfo);
//	ui->send->setEnabled(m_allGood);
}

Secret Transact::findSecret(u256 _totalReq) const
{
	if (!ethereum())
		return Secret();

	Address best;
	u256 bestBalance = 0;
	for (auto const& i: m_accounts)
	{
		auto b = ethereum()->balanceAt(i, PendingBlock);
		if (b >= _totalReq)
		{
			best = i;
			break;
		}
		if (b > bestBalance)
			bestBalance = b, best = i;
	}
	return m_context->retrieveSecret(best);
}

Address Transact::fromAccount()
{
	if (ui->from->currentIndex() < 0 || ui->from->currentIndex() >= (int)m_accounts.size())
		return Address();
	auto it = m_accounts.begin();
	std::advance(it, ui->from->currentIndex());
	return *it;
}

void Transact::updateNonce()
{
	u256 n = ethereum()->countAt(fromAccount(), PendingBlock);
	ui->nonce->setMaximum((unsigned)n);
	ui->nonce->setMinimum(0);
	ui->nonce->setValue((unsigned)n);
}

void Transact::on_send_clicked()
{
//	Secret s = findSecret(value() + fee());
	u256 nonce = ui->autoNonce->isChecked() ? ethereum()->countAt(fromAccount(), PendingBlock) : ui->nonce->value();
	auto a = fromAccount();
	auto b = ethereum()->balanceAt(a, PendingBlock);

	if (!a || b < value() + fee())
	{
		QMessageBox::critical(nullptr, "Transaction Failed", "Couldn't make transaction: account doesn't contain at least the required amount.", QMessageBox::Ok);
		return;
	}

	Secret s = m_context->retrieveSecret(a);
	if (!s)
		return;

	if (isCreation())
	{
		// If execution is a contract creation, add Natspec to
		// a local Natspec LEVELDB
		ethereum()->submitTransaction(s, value(), m_data, ui->gas->value(), gasPrice(), nonce);
#if ETH_SOLIDITY
		string src = ui->data->toPlainText().toStdString();
		if (sourceIsSolidity(src))
			try
			{
				dev::solidity::CompilerStack compiler(true);
				m_data = compiler.compile(src, ui->optimize->isChecked());
				for (string const& s: compiler.getContractNames())
				{
					h256 contractHash = compiler.getContractCodeHash(s);
					m_natSpecDB->add(contractHash, compiler.getMetadata(s, dev::solidity::DocumentationType::NatspecUser));
				}
			}
			catch (...) {}
#endif
	}
	else
		// TODO: cache like m_data.
		ethereum()->submitTransaction(s, value(), m_context->fromString(ui->destination->currentText().toStdString()).first, m_data, ui->gas->value(), gasPrice(), nonce);
	close();
}

void Transact::on_debug_clicked()
{
//	Secret s = findSecret(value() + fee());
	Address from = fromAccount();
	auto b = ethereum()->balanceAt(from, PendingBlock);
	if (!from || b < value() + fee())
	{
		QMessageBox::critical(this, "Transaction Failed", "Couldn't make transaction: account doesn't contain at least the required amount.");
		return;
	}

	try
	{
		State st(ethereum()->postState());
		Transaction t = isCreation() ?
			Transaction(value(), gasPrice(), ui->gas->value(), m_data, st.transactionsFrom(from)) :
			Transaction(value(), gasPrice(), ui->gas->value(), m_context->fromString(ui->destination->currentText().toStdString()).first, m_data, st.transactionsFrom(from));
		t.forceSender(from);
		Debugger dw(m_context, this);
		Executive e(st, ethereum()->blockChain(), 0);
		dw.populate(e, t);
		dw.exec();
	}
	catch (dev::Exception const& _e)
	{
		QMessageBox::critical(this, "Transaction Failed", "Couldn't make transaction. Low-level error: " + QString::fromStdString(diagnostic_information(_e)));
		// this output is aimed at developers, reconsider using _e.what for more user friendly output.
	}
}
