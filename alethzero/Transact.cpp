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

#include "Transact.h"

#include <fstream>
#include <QFileDialog>
#include <QMessageBox>
#include <liblll/Compiler.h>
#include <liblll/CodeFragment.h>
#include <libsolidity/CompilerStack.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/AST.h>
#include <libsolidity/SourceReferenceFormatter.h>
#include <libnatspec/NatspecExpressionEvaluator.h>
#include <libethereum/Client.h>
#include <libethereum/Utility.h>
#ifndef _MSC_VER
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

	initUnits(ui->gasPriceUnits);
	initUnits(ui->valueUnits);
	ui->valueUnits->setCurrentIndex(6);
	ui->gasPriceUnits->setCurrentIndex(4);
	ui->gasPrice->setValue(10);
	on_destination_currentTextChanged();
}

Transact::~Transact()
{
	delete ui;
}

void Transact::setEnvironment(QList<dev::KeyPair> _myKeys, dev::eth::Client* _eth, NatSpecFace* _natSpecDB)
{
	m_myKeys = _myKeys;
	m_ethereum = _eth;
	m_natSpecDB = _natSpecDB;
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

u256 Transact::total() const
{
	return value() + fee();
}

void Transact::updateDestination()
{
	cwatch << "updateDestination()";
	QString s;
	for (auto i: ethereum()->addresses())
		if ((s = m_context->pretty(i)).size())
			// A namereg address
			if (ui->destination->findText(s, Qt::MatchExactly | Qt::MatchCaseSensitive) == -1)
				ui->destination->addItem(s);
	for (int i = 0; i < ui->destination->count(); ++i)
		if (ui->destination->itemText(i) != "(Create Contract)" && !m_context->fromString(ui->destination->itemText(i)))
			ui->destination->removeItem(i--);
}

void Transact::updateFee()
{
	ui->fee->setText(QString("(gas sub-total: %1)").arg(formatBalance(fee()).c_str()));
	auto totalReq = total();
	ui->total->setText(QString("Total: %1").arg(formatBalance(totalReq).c_str()));

	bool ok = false;
	for (auto i: m_myKeys)
		if (ethereum()->balanceAt(i.address()) >= totalReq)
		{
			ok = true;
			break;
		}
	ui->send->setEnabled(ok);
	QPalette p = ui->total->palette();
	p.setColor(QPalette::WindowText, QColor(ok ? 0x00 : 0x80, 0x00, 0x00));
	ui->total->setPalette(p);
}

string Transact::getFunctionHashes(dev::solidity::CompilerStack const& _compiler, string const& _contractName)
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

void Transact::on_destination_currentTextChanged()
{
	if (ui->destination->currentText().size() && ui->destination->currentText() != "(Create Contract)")
		if (Address a = m_context->fromString(ui->destination->currentText()))
			ui->calculatedName->setText(m_context->render(a));
		else
			ui->calculatedName->setText("Unknown Address");
	else
		ui->calculatedName->setText("Create Contract");
	rejigData();
//	updateFee();
}

void Transact::rejigData()
{
	if (isCreation())
	{
		string src = ui->data->toPlainText().toStdString();
		vector<string> errors;
		QString lll;
		QString solidity;
		if (src.find_first_not_of("1234567890abcdefABCDEF") == string::npos && src.size() % 2 == 0)
		{
			m_data = fromHex(src);
		}
		else if (sourceIsSolidity(src))
		{
			dev::solidity::CompilerStack compiler;
			try
			{
//				compiler.addSources(dev::solidity::StandardSources);
				m_data = compiler.compile(src, ui->optimize->isChecked());
				solidity = "<h4>Solidity</h4>";
				solidity += "<pre>var " + QString::fromStdString(compiler.defaultContractName()) + " = web3.eth.contractFromAbi(" + QString::fromStdString(compiler.getInterface()).replace(QRegExp("\\s"), "").toHtmlEscaped() + ");</pre>";
				solidity += "<pre>" + QString::fromStdString(compiler.getSolidityInterface()).toHtmlEscaped() + "</pre>";
				solidity += "<pre>" + QString::fromStdString(getFunctionHashes(compiler)).toHtmlEscaped() + "</pre>";
			}
			catch (dev::Exception const& exception)
			{
				ostringstream error;
				solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler);
				solidity = "<h4>Solidity</h4><pre>" + QString::fromStdString(error.str()).toHtmlEscaped() + "</pre>";
			}
			catch (...)
			{
				solidity = "<h4>Solidity</h4><pre>Uncaught exception.</pre>";
			}
		}
#ifndef _MSC_VER
		else if (sourceIsSerpent(src))
		{
			try
			{
				m_data = dev::asBytes(::compile(src));
				for (auto& i: errors)
					i = "(LLL " + i + ")";
			}
			catch (string err)
			{
				errors.push_back("Serpent " + err);
			}
		}
#endif
		else
		{
			m_data = compileLLL(src, ui->optimize->isChecked(), &errors);
			if (errors.empty())
			{
				auto asmcode = compileLLLToAsm(src, false);
				lll = "<h4>Pre</h4><pre>" + QString::fromStdString(asmcode).toHtmlEscaped() + "</pre>";
				if (ui->optimize->isChecked())
				{
					asmcode = compileLLLToAsm(src, true);
					lll = "<h4>Opt</h4><pre>" + QString::fromStdString(asmcode).toHtmlEscaped() + "</pre>" + lll;
				}
			}
		}
		QString errs;
		if (errors.size())
		{
			errs = "<h4>Errors</h4>";
			for (auto const& i: errors)
				errs.append("<div style=\"border-left: 6px solid #c00; margin-top: 2px\">" + QString::fromStdString(i).toHtmlEscaped() + "</div>");
		}
		ui->code->setHtml(errs + lll + solidity + "<h4>Code</h4>" + QString::fromStdString(disassemble(m_data)).toHtmlEscaped() + "<h4>Hex</h4>" Div(Mono) + QString::fromStdString(toHex(m_data)) + "</div>");
		ui->gas->setMinimum((qint64)Interface::txGas(m_data, 0));
		if (!ui->gas->isEnabled())
			ui->gas->setValue(m_backupGas);
		ui->gas->setEnabled(true);
	}
	else
	{
		m_data = parseData(ui->data->toPlainText().toStdString());
		auto to = m_context->fromString(ui->destination->currentText());
		QString natspec;
		if (ethereum()->codeAt(to, 0).size())
		{
			string userNotice = m_natSpecDB->getUserNotice(ethereum()->postState().codeHash(to), m_data);
			if (userNotice.empty())
				natspec = "Destination contract unknown.";
			else
			{
				NatspecExpressionEvaluator evaluator;
				natspec = evaluator.evalExpression(QString::fromStdString(userNotice));
			}
			ui->gas->setMinimum((qint64)Interface::txGas(m_data, 1));
			if (!ui->gas->isEnabled())
				ui->gas->setValue(m_backupGas);
			ui->gas->setEnabled(true);
		}
		else
		{
			natspec += "Destination not a contract.";
			if (ui->gas->isEnabled())
				m_backupGas = ui->gas->value();
			ui->gas->setValue((qint64)Interface::txGas(m_data));
			ui->gas->setEnabled(false);
		}
		ui->code->setHtml("<h3>NatSpec</h3>" + natspec + "<h3>Dump</h3>" + QString::fromStdString(dev::memDump(m_data, 8, true)) + "<h3>Hex</h3>" + Div(Mono) + QString::fromStdString(toHex(m_data)) + "</div>");
	}
	updateFee();
}

void Transact::on_send_clicked()
{
	u256 totalReq = value() + fee();
	for (auto const& i: m_myKeys)
		if (ethereum()->balanceAt(i.address(), 0) >= totalReq)
		{
			Secret s = i.secret();
			if (isCreation())
			{
				// If execution is a contract creation, add Natspec to
				// a local Natspec LEVELDB
				ethereum()->transact(s, value(), m_data, ui->gas->value(), gasPrice());
				string src = ui->data->toPlainText().toStdString();
				if (sourceIsSolidity(src))
					try
					{
						dev::solidity::CompilerStack compiler;
						m_data = compiler.compile(src, ui->optimize->isChecked());
						for (string const& s: compiler.getContractNames())
						{
							h256 contractHash = compiler.getContractCodeHash(s);
							m_natSpecDB->add(contractHash, compiler.getMetadata(s, dev::solidity::DocumentationType::NatspecUser));
						}
					}
					catch (...)
					{
					}
				close();
				return;
			}
			else
				ethereum()->transact(s, value(), m_context->fromString(ui->destination->currentText()), m_data, ui->gas->value(), gasPrice());
			return;
		}
	QMessageBox::critical(this, "Transaction Failed", "Couldn't make transaction: no single account contains at least the required amount.");
}

void Transact::on_debug_clicked()
{
	try
	{
		u256 totalReq = value() + fee();
		for (auto i: m_myKeys)
			if (ethereum()->balanceAt(i.address()) >= totalReq)
			{
				State st(ethereum()->postState());
				Secret s = i.secret();
				Transaction t = isCreation() ?
					Transaction(value(), gasPrice(), ui->gas->value(), m_data, st.transactionsFrom(dev::toAddress(s)), s) :
					Transaction(value(), gasPrice(), ui->gas->value(), m_context->fromString(ui->destination->currentText()), m_data, st.transactionsFrom(dev::toAddress(s)), s);
				Debugger dw(m_context, this);
				Executive e(st, ethereum()->blockChain(), 0);
				dw.populate(e, t);
				dw.exec();
				return;
			}
			QMessageBox::critical(this, "Transaction Failed", "Couldn't make transaction: no single account contains at least the required amount.");
	}
	catch (dev::Exception const& _e)
	{
		QMessageBox::critical(this, "Transaction Failed", "Couldn't make transaction. Low-level error: " + QString::fromStdString(diagnostic_information(_e)));
		// this output is aimed at developers, reconsider using _e.what for more user friendly output.
	}
}
